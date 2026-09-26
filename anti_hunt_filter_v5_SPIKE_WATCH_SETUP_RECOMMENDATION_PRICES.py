import json
import os
import sys
import time
from datetime import datetime, time as dt_time, date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# V5: bounded strategy + richer market features + walk-forward shadow ML
# ============================================================

TICKER = "ZW=F"
CHICAGO_TZ = ZoneInfo("America/Chicago")
TICK_SIZE = 0.25

ATR_PERIOD = 16
ATR_STOP_MIN_MULT = 2.0
MAX_DATA_AGE_MIN = 45
MAX_FETCH_ATTEMPTS = 3

WINDOW_OPEN = dt_time(8, 45)
WINDOW_CLOSE = dt_time(12, 30)
ENTRY_CUTOFF = dt_time(11, 30)

PROFILES = {
    "BASE": {"entry_mult": 0.994, "stop_mult": 1.012, "target_mult": 0.960},
    "ENTRY_993": {"entry_mult": 0.993, "stop_mult": 1.012, "target_mult": 0.960},
    "STOP_1010": {"entry_mult": 0.994, "stop_mult": 1.010, "target_mult": 0.960},
    "STOP_1014": {"entry_mult": 0.994, "stop_mult": 1.014, "target_mult": 0.960},
    "TARGET_958": {"entry_mult": 0.994, "stop_mult": 1.012, "target_mult": 0.958},
}
DEFAULT_PROFILE = "BASE"

MIN_LEARNING_SETUPS = 30
LEARNING_CHECKPOINT = 10
MIN_EXPECTED_R_IMPROVEMENT = 0.05
MIN_RECENT_EXPECTED_R_IMPROVEMENT = 0.02

ML_SHADOW_ONLY = True
ML_MIN_SAMPLES = 60
ML_MIN_CLASS_COUNT = 15
ML_CONFIDENCE_THRESHOLD = 0.60
ML_TEST_WINDOW = 20
ML_MIN_TRAIN_SAMPLES = 40

STATE_FILE = "learning_state.json"
OUTCOME_VERSION = 5
SETUP_FILE = "setup.json"
REPORT_FILE = "learning_report.json"
ML_REPORT_FILE = "ml_report.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEALTHCHECK_URL = os.environ.get("HEALTHCHECK_URL", "").strip()

HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}

ML_FEATURES = [
    "open_distance_pct","atr_pct","stop_atr_multiple","rr","hour_decimal",
    "stale_data_min","vol_warning","current_vs_open_pct","rsi_14",
    "vwap_distance_pct","volume_ratio","momentum_4bar_pct","trend_16bar_pct",
    "range_position","opening_range_position","prior_day_range_pct",
    "current_vs_prior_high_pct","current_vs_prior_low_pct","atr_regime_ratio",
    "trend_1h_pct","regime_score",
]

def is_manual(): return "--manual" in sys.argv
def is_resolve(): return "--resolve" in sys.argv
def is_force_stale(): return "--force-stale" in sys.argv

def ping_healthcheck():
    if not HEALTHCHECK_URL or is_manual(): return
    try: requests.get(HEALTHCHECK_URL, timeout=10)
    except requests.RequestException as exc: print(f"Healthcheck ping failed (non-fatal): {exc}")

def round_tick(price): return round(round(float(price) / TICK_SIZE) * TICK_SIZE, 4)

def chicago_index(df):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None: idx = idx.tz_localize("UTC")
    return idx.tz_convert(CHICAGO_TZ)

def check_time_window():
    now = datetime.now(CHICAGO_TZ)
    return now.weekday() < 5 and WINDOW_OPEN <= now.time() <= WINDOW_CLOSE

def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy(); df.columns = df.columns.get_level_values(0)
    return df

def fetch_market_data():
    last_error = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            intraday = _flatten(yf.download(TICKER, period="2d", interval="15m", auto_adjust=False, progress=False))
            daily = _flatten(yf.download(TICKER, period="5d", interval="1d", auto_adjust=False, progress=False))
            hourly = _flatten(yf.download(TICKER, period="5d", interval="60m", auto_adjust=False, progress=False))
            if not intraday.empty and not daily.empty and not hourly.empty:
                return intraday, daily, hourly
            last_error = "Yahoo returned empty data."
        except Exception as exc:
            last_error = str(exc); print(f"Fetch attempt {attempt} failed: {exc}", file=sys.stderr)
        if attempt < MAX_FETCH_ATTEMPTS: time.sleep(10 * attempt)
    raise ValueError(f"Could not fetch usable ZW=F data: {last_error}")

def resolve_session_open(intraday, daily):
    now_ct = datetime.now(CHICAGO_TZ)
    idx = chicago_index(intraday)
    todays = intraday.loc[[ts.date() == now_ct.date() for ts in idx]]
    return float(todays["Open"].iloc[0]) if not todays.empty else float(daily["Open"].iloc[-1])

def compute_atr(intraday):
    high, low, close = intraday["High"].astype(float), intraday["Low"].astype(float), intraday["Close"].astype(float)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    value = tr.rolling(ATR_PERIOD).mean().iloc[-1]
    return float(value) if pd.notna(value) else None

def check_staleness(intraday):
    idx = chicago_index(intraday)
    return max(0.0, (datetime.now(CHICAGO_TZ) - idx[-1]).total_seconds()/60.0)

def load_state():
    default = {"version":5,"active_profile":DEFAULT_PROFILE,"setups":[],"last_learning_update":None,"learning_updates":[]}
    if not os.path.exists(STATE_FILE): return default
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh: state = json.load(fh)
        if not isinstance(state, dict): return default
        state.setdefault("version",5); state["version"]=5
        state.setdefault("active_profile",DEFAULT_PROFILE); state.setdefault("setups",[])
        state.setdefault("last_learning_update",None); state.setdefault("learning_updates",[])
        if state["active_profile"] not in PROFILES: state["active_profile"]=DEFAULT_PROFILE
        return state
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Learning state unreadable; using safe defaults: {exc}"); return default

def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)): return value.isoformat()
    if value is pd.NaT: return None
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, (str,int,float,bool)) or value is None: return value
    return str(value)

def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh: json.dump(json_safe(data), fh, indent=2)
    os.replace(tmp, path)

def _safe_float(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default

def rsi(series, period=14):
    delta=series.astype(float).diff(); gain=delta.clip(lower=0).rolling(period).mean()
    loss=(-delta.clip(upper=0)).rolling(period).mean(); rs=gain/loss.replace(0,np.nan)
    return (100-(100/(1+rs))).fillna(50.0)

def market_context(intraday, daily, hourly, now_ct):
    df=intraday.copy(); df.index=chicago_index(df); df=df.sort_index()
    today=df.loc[df.index.date==now_ct.date()].copy()
    if today.empty: today=df.tail(32).copy()
    close,high,low=df["Close"].astype(float),df["High"].astype(float),df["Low"].astype(float)
    volume=df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1.0,index=df.index)
    latest=float(close.iloc[-1]); rsi14=float(rsi(close,14).iloc[-1])
    momentum_4=(latest/float(close.iloc[-5])-1)*100 if len(close)>=5 else 0
    trend_16=(latest/float(close.iloc[-17])-1)*100 if len(close)>=17 else 0
    recent_vol=volume.tail(20).replace([np.inf,-np.inf],np.nan).dropna()
    vol_ratio=float(volume.iloc[-1])/float(recent_vol.median()) if not recent_vol.empty and recent_vol.median()>0 else 1
    typical=(today["High"].astype(float)+today["Low"].astype(float)+today["Close"].astype(float))/3
    tv=today["Volume"].astype(float) if "Volume" in today.columns else pd.Series(1.0,index=today.index)
    denom=float(tv.sum()); vwap=float((typical*tv).sum()/denom) if denom>0 else latest
    vwap_distance=(latest-vwap)/vwap*100 if vwap else 0
    sh,sl=float(today["High"].astype(float).max()),float(today["Low"].astype(float).min()); span=sh-sl
    range_position=(latest-sl)/span if span>0 else .5
    opening=today.head(4); oh=float(opening["High"].astype(float).max()) if not opening.empty else latest
    ol=float(opening["Low"].astype(float).min()) if not opening.empty else latest; os_=oh-ol
    opening_position=(latest-ol)/os_ if os_>0 else .5
    prior=daily.iloc[-2] if len(daily)>=2 else daily.iloc[-1]
    ph=_safe_float(prior.get("High"),latest); pl=_safe_float(prior.get("Low"),latest); po=_safe_float(prior.get("Open"),latest)
    prior_range_pct=(ph-pl)/po*100 if po else 0; vs_ph=(latest-ph)/ph*100 if ph else 0; vs_pl=(latest-pl)/pl*100 if pl else 0
    atr_series=pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    atr_now=_safe_float(atr_series.iloc[-1],0); atr_short=_safe_float(atr_series.tail(8).mean(),atr_now)
    atr_regime=atr_now/atr_short if atr_short>0 else 1
    h=hourly.copy(); h.index=chicago_index(h); hclose=h["Close"].astype(float)
    trend_1h=(float(hclose.iloc[-1])/float(hclose.iloc[-4])-1)*100 if len(hclose)>=4 else 0
    trend_score=1 if trend_16>.20 else -1 if trend_16<-.20 else 0
    regime="HIGH_VOL" if atr_regime>1.35 else "LOW_VOL" if atr_regime<.75 else "TREND_UP" if trend_score>0 else "TREND_DOWN" if trend_score<0 else "RANGE"
    return {"rsi_14":round(rsi14,4),"vwap":round(vwap,4),"vwap_distance_pct":round(vwap_distance,6),"volume_ratio":round(vol_ratio,4),
            "momentum_4bar_pct":round(momentum_4,6),"trend_16bar_pct":round(trend_16,6),"range_position":round(range_position,6),
            "opening_range_position":round(opening_position,6),"prior_day_range_pct":round(prior_range_pct,6),
            "current_vs_prior_high_pct":round(vs_ph,6),"current_vs_prior_low_pct":round(vs_pl,6),"atr_regime_ratio":round(atr_regime,6),
            "trend_1h_pct":round(trend_1h,6),"regime":regime,"regime_score":trend_score}

def build_setup(daily_open,current_price,atr,profile_name):
    p=PROFILES[profile_name]; entry=round_tick(daily_open*p["entry_mult"]); stop=round_tick(daily_open*p["stop_mult"]); target=round_tick(daily_open*p["target_mult"])
    risk=round(stop-entry,4); reward=round(entry-target,4); now=datetime.now(CHICAGO_TZ)
    return {"ticker":TICKER,"profile":profile_name,"timestamp_ct":now.isoformat(),
            "valid_until_ct":datetime.combine(now.date(),WINDOW_CLOSE,tzinfo=CHICAGO_TZ).isoformat(),
            "daily_open":round(daily_open,4),"current_price":round(current_price,4),"entry":entry,"stop":stop,"target":target,
            "risk":risk,"reward":reward,"rr":round(reward/risk,2) if risk>0 else 0.0,"atr_15m":round(atr,4) if atr else None,
            "stop_atr_multiple":round(risk/atr,2) if atr and atr>0 else None,"vol_warning":bool(atr and risk<ATR_STOP_MIN_MULT*atr),
            "invalidated":current_price>stop}

def feature_vector(setup):
    ts=datetime.fromisoformat(setup["timestamp_ct"]); daily_open=float(setup["daily_open"]); current=float(setup["current_price"]); atr=float(setup["atr_15m"]) if setup.get("atr_15m") else 0
    return [(float(setup["entry"])-daily_open)/daily_open*100 if daily_open else 0,(atr/daily_open*100) if daily_open else 0,
            float(setup["stop_atr_multiple"] or 0),float(setup["rr"]),ts.hour+ts.minute/60,float(setup.get("stale_data_min",0)),
            1.0 if setup.get("vol_warning") else 0,(current-daily_open)/daily_open*100 if daily_open else 0,
            *[_safe_float(setup.get(name),0) for name in ML_FEATURES[8:]]]

def historical_ml_rows(state):
    rows=[]
    for setup in state.get("setups",[]):
        if setup.get("actual_outcome") not in {"WIN","LOSS"} or not setup.get("market_context"): continue
        try: rows.append({"timestamp":setup.get("timestamp_ct",""),"features":feature_vector(setup),"label":1 if setup["actual_outcome"]=="WIN" else 0})
        except (KeyError,TypeError,ValueError): continue
    rows.sort(key=lambda x:x["timestamp"]); return rows

def make_logistic():
    return Pipeline([("scale",StandardScaler()),("logistic",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42))])

def make_boosting():
    return HistGradientBoostingClassifier(max_iter=120,learning_rate=.05,max_leaf_nodes=7,l2_regularization=1.0,random_state=42)

def train_ml_model(state):
    rows=historical_ml_rows(state); n=len(rows)
    if n<ML_MIN_SAMPLES: return None,{"trained":False,"reason":f"Need at least {ML_MIN_SAMPLES} resolved WIN/LOSS samples.","samples":n}
    y=[r["label"] for r in rows]
    if len(set(y))<2 or min(y.count(0),y.count(1))<ML_MIN_CLASS_COUNT: return None,{"trained":False,"reason":f"Both WIN and LOSS classes need at least {ML_MIN_CLASS_COUNT} samples.","samples":n,"wins":y.count(1),"losses":y.count(0)}
    X=[r["features"] for r in rows]; test_n=min(ML_TEST_WINDOW,max(10,n//5)); split=n-test_n
    if split<ML_MIN_TRAIN_SAMPLES or len(set(y[:split]))<2: return None,{"trained":False,"reason":"Not enough earlier time-ordered data for a walk-forward test.","samples":n,"test_window":test_n}
    evaluations=[]
    for name,candidate in [("logistic",make_logistic()),("boosting",make_boosting())]:
        candidate.fit(X[:split],y[:split]); probs=candidate.predict_proba(X[split:])[:,1]; preds=(probs>=ML_CONFIDENCE_THRESHOLD).astype(int); actual=y[split:]
        ev={"model":name,"accuracy_at_threshold":round(float(accuracy_score(actual,preds)),4),"brier_score":round(float(brier_score_loss(actual,probs)),6),"log_loss":round(float(log_loss(actual,np.clip(probs,1e-6,1-1e-6))),6)}
        high=[i for i,p in enumerate(probs) if p>=ML_CONFIDENCE_THRESHOLD or p<=1-ML_CONFIDENCE_THRESHOLD]; ev["high_confidence_samples"]=len(high)
        ev["high_confidence_accuracy"]=round(sum(int((probs[i]>=ML_CONFIDENCE_THRESHOLD)==bool(actual[i])) for i in high)/len(high),4) if high else None; evaluations.append(ev)
    selected_name=sorted(evaluations,key=lambda x:(x["brier_score"],x["log_loss"]))[0]["model"]; selected=make_logistic() if selected_name=="logistic" else make_boosting(); selected.fit(X,y)
    return selected,{"trained":True,"samples":n,"wins":y.count(1),"losses":y.count(0),"walk_forward_test_window":test_n,"walk_forward_train_samples":split,"candidate_models":evaluations,"selected_model":selected_name,"selection_rule":"lowest walk-forward Brier score, then log loss","shadow_only":ML_SHADOW_ONLY}

def add_ml_prediction(state,setup):
    model,report=train_ml_model(state); setup["ml_shadow"]={"enabled":True,"trained":bool(model),"probability_win":None,"confidence_threshold":ML_CONFIDENCE_THRESHOLD,"decision":"INSUFFICIENT_DATA","model_samples":report.get("samples",0),"selected_model":report.get("selected_model")}
    if model is not None:
        probability=float(model.predict_proba([feature_vector(setup)])[0][1]); setup["ml_shadow"].update({"probability_win":round(probability,4),"decision":"WATCH" if probability>=ML_CONFIDENCE_THRESHOLD else "LOW_CONFIDENCE"})
    return report

def completed_bars(intraday):
    idx=chicago_index(intraday); now=datetime.now(CHICAGO_TZ); mask=[(ts+timedelta(minutes=15))<=now for ts in idx]
    return intraday.loc[mask].copy(),idx[mask]

def evaluate_geometry(setup,bars,idx,profile_name):
    daily_open=float(setup["daily_open"]); valid_until=datetime.fromisoformat(setup["valid_until_ct"]); setup_time=datetime.fromisoformat(setup["timestamp_ct"])
    p=PROFILES[profile_name]; entry=round_tick(daily_open*p["entry_mult"]); stop=round_tick(daily_open*p["stop_mult"]); target=round_tick(daily_open*p["target_mult"]); risk=stop-entry; reward=entry-target
    entered=False; entry_time=None; max_favorable=0.0; max_adverse=0.0
    for i,(_,bar) in enumerate(bars.iterrows()):
        bar_time=idx[i]
        if bar_time<=setup_time: continue
        if bar_time>valid_until: break
        high,low=float(bar["High"]),float(bar["Low"])
        if not entered and low<=entry<=high: entered=True; entry_time=bar_time
        if entered:
            max_favorable=max(max_favorable,(entry-low)/risk if risk>0 else 0); max_adverse=max(max_adverse,(high-entry)/risk if risk>0 else 0)
            if high>=stop and low<=target: return {"outcome":"AMBIGUOUS","r_multiple":None,"entry_time_ct":entry_time.isoformat(),"mfe_r":round(max_favorable,4),"mae_r":round(max_adverse,4),"bars_after_entry":i+1}
            if high>=stop: return {"outcome":"LOSS","r_multiple":-1.0,"entry_time_ct":entry_time.isoformat(),"mfe_r":round(max_favorable,4),"mae_r":round(max_adverse,4),"bars_after_entry":i+1}
            if low<=target: return {"outcome":"WIN","r_multiple":round(reward/risk,4),"entry_time_ct":entry_time.isoformat(),"mfe_r":round(max_favorable,4),"mae_r":round(max_adverse,4),"bars_after_entry":i+1}
    if not entered: return {"outcome":"NO_ENTRY","r_multiple":0.0,"entry_time_ct":None,"mfe_r":0.0,"mae_r":0.0,"bars_after_entry":0}
    return {"outcome":"EXPIRED_AFTER_ENTRY","r_multiple":0.0,"entry_time_ct":entry_time.isoformat(),"mfe_r":round(max_favorable,4),"mae_r":round(max_adverse,4),"bars_after_entry":len(bars)}

def resolve_previous_setups(state,intraday):
    bars,idx=completed_bars(intraday)
    if bars.empty: return 0
    changed=0
    for setup in state.get("setups",[]):
        if setup.get("resolved_profiles"): continue
        try:
            valid_until=datetime.fromisoformat(setup["valid_until_ct"])
            if datetime.now(CHICAGO_TZ)<=valid_until: continue
        except (KeyError,ValueError): continue
        evaluations={name:evaluate_geometry(setup,bars,idx,name) for name in PROFILES}
        setup["resolved_profiles"]=evaluations; actual=evaluations.get(setup.get("profile",DEFAULT_PROFILE))
        if actual:
            setup["actual_outcome"]=actual["outcome"]; setup["actual_r_multiple"]=actual.get("r_multiple"); setup["actual_entry_time_ct"]=actual.get("entry_time_ct")
            setup["actual_mfe_r"]=actual.get("mfe_r"); setup["actual_mae_r"]=actual.get("mae_r"); setup["actual_bars_after_entry"]=actual.get("bars_after_entry"); setup["outcome_version"]=OUTCOME_VERSION
        changed+=1
    return changed

def profile_stats(state):
    stats={}
    for profile_name in PROFILES:
        evaluations=[(s.get("resolved_profiles") or {}).get(profile_name) for s in state.get("setups",[])]; evaluations=[x for x in evaluations if x]
        trades=[x for x in evaluations if x["outcome"] in {"WIN","LOSS"}]; entered=[x for x in evaluations if x["outcome"] in {"WIN","LOSS","AMBIGUOUS","EXPIRED_AFTER_ENTRY"}]
        r_values=[float(x["r_multiple"]) for x in evaluations if x.get("r_multiple") is not None]; recent=evaluations[-20:]; recent_r=[float(x["r_multiple"]) for x in recent if x.get("r_multiple") is not None]; wins=sum(x["outcome"]=="WIN" for x in trades)
        stats[profile_name]={"setups":len(evaluations),"trades":len(trades),"wins":wins,"losses":sum(x["outcome"]=="LOSS" for x in trades),"entries":len(entered),"no_entry":sum(x["outcome"]=="NO_ENTRY" for x in evaluations),"ambiguous":sum(x["outcome"]=="AMBIGUOUS" for x in evaluations),"expired_after_entry":sum(x["outcome"]=="EXPIRED_AFTER_ENTRY" for x in evaluations),"win_rate":round(wins/len(trades),4) if trades else None,"expected_r":round(sum(r_values)/len(evaluations),4) if evaluations else None,"avg_r_per_trade":round(sum(r_values)/len(trades),4) if trades else None,"recent_expected_r":round(sum(recent_r)/len(recent),4) if recent else None}
    return stats

def maybe_learn(state):
    live_setups=[s for s in state.get("setups",[]) if s.get("source")!="backtest"]; stats=profile_stats({"setups":live_setups})
    report={"generated_at_ct":datetime.now(CHICAGO_TZ).isoformat(),"active_profile":state["active_profile"],"profiles":stats,"learning_applied":False,"reason":"Not enough completed setups."}
    completed_count=max((x["setups"] for x in stats.values()),default=0)
    if completed_count<MIN_LEARNING_SETUPS: return report
    if completed_count%LEARNING_CHECKPOINT!=0: report["reason"]="Not at a learning checkpoint."; return report
    current_name=state["active_profile"]; current=stats.get(current_name,{})
    if current.get("recent_expected_r") is None: report["reason"]="Current profile has insufficient recent data."; return report
    candidates=[(d["recent_expected_r"],d["expected_r"],name) for name,d in stats.items() if d["setups"]>=MIN_LEARNING_SETUPS and d["recent_expected_r"] is not None]
    if not candidates: report["reason"]="No profile has enough observations."; return report
    candidates.sort(reverse=True); best_recent,best_overall,best_name=candidates[0]
    if best_name==current_name: report["reason"]="Current profile remains the best observed profile."; return report
    improvement_recent=best_recent-current["recent_expected_r"]; improvement_overall=best_overall-(current.get("expected_r") or 0)
    if improvement_recent<MIN_RECENT_EXPECTED_R_IMPROVEMENT: report["reason"]="Recent improvement is below the safety threshold."; return report
    if improvement_overall<MIN_EXPECTED_R_IMPROVEMENT: report["reason"]="Overall improvement is below the safety threshold."; return report
    previous=state["active_profile"]; state["active_profile"]=best_name; state["last_learning_update"]=datetime.now(CHICAGO_TZ).isoformat()
    state["learning_updates"].append({"timestamp_ct":state["last_learning_update"],"previous_profile":previous,"new_profile":best_name,"previous_recent_expected_r":current["recent_expected_r"],"new_recent_expected_r":best_recent,"previous_expected_r":current.get("expected_r"),"new_expected_r":best_overall})
    report.update({"learning_applied":True,"reason":"New bounded profile passed recent and overall improvement thresholds.","previous_profile":previous,"new_profile":best_name,"improvement_recent_expected_r":round(improvement_recent,4),"improvement_overall_expected_r":round(improvement_overall,4)})
    return report

def ml_shadow_report(state):
    model,report=train_ml_model(state); report["generated_at_ct"]=datetime.now(CHICAGO_TZ).isoformat(); report["model_type"]="Walk-forward LogisticRegression / HistGradientBoosting"; report["features"]=ML_FEATURES; report["confidence_threshold"]=ML_CONFIDENCE_THRESHOLD; report["shadow_only"]=ML_SHADOW_ONLY
    return report

def send_telegram_alert(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing; alert printed instead."); print(text); return False
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"; payload={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}
    try:
        response=requests.post(url,json=payload,timeout=15); response.raise_for_status(); return True
    except requests.RequestException as exc:
        print(f"Telegram delivery failed (non-fatal): {exc}"); print(text); return False

def format_message(setup,report):
    lines=["⛓ <b>[ZW=F] Anti-Hunt Short Setup V5</b>","━━━━━━━━━━━━━━━━━━━━",
           f"📅 {datetime.now(CHICAGO_TZ).strftime('%A %Y-%m-%d %H:%M %Z')}",f"🧠 Profile: <b>{setup['profile']}</b>",
           f"🔵 Open: <b>{setup['daily_open']}</b>",f"📊 Current: <code>{setup['current_price']}</code>","━━━━━━━━━━━━━━━━━━━━",
           f"🔻 ENTRY: <code>{setup['entry']}</code>",f"🛑 STOP: <code>{setup['stop']}</code>",f"🎯 TARGET: <code>{setup['target']}</code>",
           f"Risk: <code>{setup['risk']}</code> | Reward: <code>{setup['reward']}</code> | R:R <code>{setup['rr']}</code>"]
    ml=setup.get("ml_shadow",{})
    lines.append(f"🤖 ML SHADOW: <code>{ml['probability_win']:.1%}</code> target-before-stop | <b>{ml['decision']}</b>" if ml.get("probability_win") is not None else "🤖 ML SHADOW: collecting training data")
    if setup.get("regime"): lines.append(f"🌡 Regime: <code>{setup['regime']}</code> | RSI <code>{setup.get('rsi_14',0):.1f}</code> | VWAP dist <code>{setup.get('vwap_distance_pct',0):.2f}%</code>")
    if setup.get("stop_atr_multiple") is not None: lines.append(f"ATR: <code>{setup['atr_15m']}</code> | Stop distance: <code>{setup['stop_atr_multiple']}x</code>")
    if setup["vol_warning"]: lines.append("⚠️ Stop is inside the 2x ATR noise band.")
    if setup.get("stale_data_min",0)>MAX_DATA_AGE_MIN: lines.append(f"⚠️ STALE DATA: {setup['stale_data_min']} min")
    if setup.get("manual_mode"): lines.append("🧪 <b>MANUAL TEST ALERT</b> — stale-data protection bypassed for manual run.")
    if report.get("learning_applied"): lines.append(f"🧠 Bounded profile changed: {report['previous_profile']} → {report['new_profile']}")
    lines += ["⏳ Valid until 12:30 CT.","🤖 ML is SHADOW-ONLY in V5; it does not change the signal."]
    return "\n".join(lines)

def main():
    manual=is_manual(); resolve_only=is_resolve() and not manual; now_ct=datetime.now(CHICAGO_TZ)
    if resolve_only:
        ping_healthcheck()
        if now_ct.date() in HOLIDAYS or now_ct.weekday()>=5: print("No weekday trading session to resolve."); return 0
        print("="*50); print("OUTCOME RESOLUTION + ML LEARNING MODE"); print("No new setup will be created."); print("="*50)
    elif not manual:
        ping_healthcheck()
        if now_ct.date() in HOLIDAYS: print(f"{now_ct.date()} is a CME holiday. Aborting."); return 0
        if not check_time_window(): print(f"[{now_ct:%Y-%m-%d %H:%M %Z}] Outside execution window. Aborting."); return 0
        if now_ct.time()>ENTRY_CUTOFF: print("Past entry cutoff. Aborting."); return 0
    if manual:
        print("="*50); print("MANUAL TEST MODE"); print("Normal time/holiday restrictions are bypassed."); print("Real ZW=F data will still be fetched."); print("="*50)
    print("Fetching ZW=F data...")
    try: intraday,daily,hourly=fetch_market_data()
    except ValueError as exc: print(f"Data error: {exc}",file=sys.stderr); return 1
    state=load_state(); resolved=resolve_previous_setups(state,intraday); print(f"Resolved {resolved} previous setup(s).")
    report=maybe_learn(state); ml_report=ml_shadow_report(state); save_json(REPORT_FILE,report); save_json(ML_REPORT_FILE,ml_report)
    if resolve_only:
        save_json(STATE_FILE,state); print(json.dumps(json_safe(report),indent=2)); print(json.dumps(json_safe(ml_report),indent=2)); return 0
    age=check_staleness(intraday)
    if age>MAX_DATA_AGE_MIN:
        print(f"WARNING: last 15m bar is {age:.0f} min old (max allowed {MAX_DATA_AGE_MIN} min).")
        if not manual:
            print("ABORT: data too stale — no setup will be built, no alert sent.")
            save_json(STATE_FILE,state); return 0
        print("MANUAL MODE: continuing with stale data so the manual alert can be sent.")
    daily_open=resolve_session_open(intraday,daily); current_price=float(intraday["Close"].iloc[-1]); atr=compute_atr(intraday)
    setup=build_setup(daily_open,current_price,atr,state["active_profile"]); setup["manual_mode"]=manual; setup["stale_data_min"]=round(age,1)
    context=market_context(intraday,daily,hourly,now_ct); setup["market_context"]=context; setup.update(context)
    ml_training_report=add_ml_prediction(state,setup); setup["ml_shadow"]["training_report"]=ml_training_report
    print(json.dumps(json_safe(setup),indent=2))
    if setup["invalidated"]:
        print("Setup invalidated; no alert."); save_json(STATE_FILE,state); save_json(SETUP_FILE,setup); return 0
    if not manual: state["setups"].append(setup)
    save_json(SETUP_FILE,setup); save_json(STATE_FILE,state)
    send_telegram_alert(format_message(setup,report))
    print(json.dumps(json_safe(report),indent=2)); print(json.dumps(json_safe(ml_report),indent=2)); return 0

if __name__=="__main__": sys.exit(main())
