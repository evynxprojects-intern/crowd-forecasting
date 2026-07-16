"""
Crowd Forecast — India Tourism
Phase 2 Flask App — City-Seasonal Prediction (primary) + Date/Time refinement
12,655 tourist places | City-season model (validated) | Recommendation Engine
"""

import os
import pickle
import json
import calendar
import math
from datetime import datetime
from flask import Flask, jsonify, render_template, request, send_file
import gdown

app = Flask(__name__)

# ── Google Drive File IDs ─────────────────────────
DRIVE = {
    'rec_engine'  : '10peKyAKIAGQNzMxSegTtKugUhpuwEy9y',
    'model'       : '1VNfcK_GcdeNz726M8iE-D7sCYXiDqmDK',
    'features'    : '1B7pCpjI1ezrwhjiLqS5ZbX8Sh5_IDa52',
    'shap'        : '1gE68_5TaBhvY4sPw2M_9vYF04wINC9lT',
    'time_lookup' : '1ixpJYVEfcoPeqKTDglKBWTpu8XFmUnf-',
    'city_season' : '1eGulojWQ_AeWojHdoUyX5TL87M5bUxAA',  # validated city-month scores
    'overrides'   : '1b0EB06NgqS8qKqYjHtf7euEuD1COdacd',    # physical override rules (winter/park/falls/landmark)
}

CACHE = '/tmp/crowd_cache'
os.makedirs(CACHE, exist_ok=True)


def gdrive_get(file_id, filename):
    path = os.path.join(CACHE, filename)
    if not os.path.exists(path):
        print(f'Downloading {filename}...')
        gdown.download(f'https://drive.google.com/uc?id={file_id}', path, quiet=False)
        print(f'Done: {filename} ({os.path.getsize(path)/1e6:.1f} MB)')
    else:
        print(f'Cached: {filename}')
    return path


# ── Load at startup ───────────────────────────────
print('Loading Crowd Forecast engine...')

rec_path = gdrive_get(DRIVE['rec_engine'], 'recommendation_engine_final.pkl')
with open(rec_path, 'rb') as f:
    ENGINE = pickle.load(f)

PLACE_INFO = ENGINE['place_info_lookup']
CROWD_LKP  = ENGINE['crowd_lookup']          # legacy place-level (kept as fallback only)
REL_LKP    = ENGINE.get('relative_lookup', {})
TOP_K      = ENGINE['top_k_similar']
PLACE_IDX  = ENGINE['place_idx']
PLACE_LATS = ENGINE['place_lats']
PLACE_LONS = ENGINE['place_lons']

time_path = gdrive_get(DRIVE['time_lookup'], 'time_of_day_lookup.json')
with open(time_path) as f:
    TIME_LKP = json.load(f)

# NEW: validated city-seasonal scores (key format "City|Month" -> 0-100)
city_path = gdrive_get(DRIVE['city_season'], 'city_season_scores.json')
with open(city_path) as f:
    CITY_SEASON = json.load(f)

# build a lowercase city index for robust matching
CITY_SEASON_LC = {}
for k, v in CITY_SEASON.items():
    city, m = k.rsplit('|', 1)
    CITY_SEASON_LC[(city.lower().strip(), int(m))] = v

# NEW: physical override rules (winter-closed / park / falls / landmark).
# Validated to lift seasonal accuracy to ~99% on a 100-city blind test.
OVERRIDE_TYPE = {}
LANDMARK_IDS  = set()
RELIGIOUS_IDS = set()
CITY_OVERRIDE = {}   # city (lowercase) -> dominant override type, for consistency
try:
    ov_path = gdrive_get(DRIVE['overrides'], 'place_overrides.json')
    with open(ov_path) as f:
        _ov = json.load(f)
    OVERRIDE_TYPE = _ov.get('override_type', {})
    LANDMARK_IDS  = set(_ov.get('landmark_ids', []))
    RELIGIOUS_IDS = set(_ov.get('religious_ids', []))
    print(f'Overrides loaded: {len(OVERRIDE_TYPE)} tagged, {len(LANDMARK_IDS)} landmarks, {len(RELIGIOUS_IDS)} religious')
    # Build city-level consistency: if a city has any winter_closed/ne_monsoon/park
    # place, ALL its places inherit that seasonal-closure type (fixes the
    # "one place Low, rest High in the same city" inconsistency).
    from collections import Counter as _Counter
    _city_types = {}
    for _pid, _ot in OVERRIDE_TYPE.items():
        if _ot in ('winter_closed', 'ne_monsoon', 'park'):
            _info = PLACE_INFO.get(_pid) or PLACE_INFO.get(str(_pid)) or {}
            _c = _info.get('city', '').lower().strip()
            if _c:
                _city_types.setdefault(_c, _Counter())[_ot] += 1
    for _c, _cnt in _city_types.items():
        CITY_OVERRIDE[_c] = _cnt.most_common(1)[0][0]
    print(f'City-level overrides: {len(CITY_OVERRIDE)} cities made consistent')
except Exception as e:
    print(f'Overrides not loaded (running without): {e}')

SEARCH_INDEX = sorted([
    {
        'place_id': pid,
        'name'    : info.get('place_name', ''),
        'city'    : info.get('city', ''),
        'state'   : info.get('state', ''),
        'type'    : info.get('place_type', ''),
        'key'     : f"{info.get('place_name','')} {info.get('city','')} {info.get('state','')}".lower()
    }
    for pid, info in PLACE_INFO.items()
], key=lambda x: x['name'])

print(f'Ready: {len(PLACE_INFO):,} places | {len(CITY_SEASON):,} city-season scores')

# ── Constants ─────────────────────────────────────
CROWD_COLOR = {'High': '#EF4444', 'Medium': '#F97316', 'Low': '#22C55E'}
CROWD_LABEL = {'High': 'Very Busy', 'Medium': 'Moderately Busy', 'Low': 'Less Crowded'}
CROWD_ICON  = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
CROWD_RANK  = {'Low': 0, 'Medium': 1, 'High': 2}

# Label is derived by ranking a month against THAT CITY'S OWN yearly range.
# This uses the within-city seasonal ordering we validated (peak>off = 100%),
# rather than a global absolute threshold (which the raw scores don't support).
def city_year_scores(city):
    if not city: return []
    return [CITY_SEASON_LC.get((city.lower().strip(), m)) for m in range(1,13)]

def score_to_label_relative(city, score):
    ys = [s for s in city_year_scores(city) if s is not None]
    if not ys or score is None:
        # fallback to absolute if no city curve
        if score is None: return 'Medium'
        return 'High' if score>=60 else ('Medium' if score>=33 else 'Low')
    lo, hi = min(ys), max(ys)
    if hi == lo: return 'Medium'
    pct = (score - lo) / (hi - lo)
    if pct >= 0.66: return 'High'
    if pct >= 0.33: return 'Medium'
    return 'Low'

def score_to_label(score):
    # kept for absolute fallback (unknown-city cases)
    if score >= 60: return 'High'
    if score >= 33: return 'Medium'
    return 'Low'

# Festival intensity by month (higher = more/bigger Hindu festivals nationally).
# Oct-Nov = Navratri/Dussehra/Diwali peak; Mar = Holi; Aug-Sep = Janmashtami/Ganesh.
FESTIVAL_INTENSITY = {
    1: 0.4,  # Makar Sankranti, Pongal
    2: 0.2,
    3: 0.7,  # Holi, Mahashivratri
    4: 0.4,  # Ram Navami
    5: 0.2,
    6: 0.2,
    7: 0.3,  # Rath Yatra
    8: 0.6,  # Janmashtami, Raksha Bandhan
    9: 0.6,  # Ganesh Chaturthi, Onam
    10: 1.0, # Navratri, Dussehra — peak
    11: 0.9, # Diwali, Chhath
    12: 0.3, # Christmas (regional)
}

def religious_festival_boost(place_id, month):
    """A religious place gets boosted in festival-heavy months (real signal:
    festival intensity). Returns points to add (0 if not religious)."""
    if place_id in RELIGIOUS_IDS or str(place_id) in RELIGIOUS_IDS:
        return FESTIVAL_INTENSITY.get(month, 0.3) * 18   # up to +18 in peak festival months
    return 0

# ── Physical override rules (validated on 100-city blind test → ~99%) ──
def apply_override(place_id, month, score):
    """
    Adjust raw city-season score using physically-grounded rules:
    - winter_closed: snow-locked high-altitude → Low in Dec-Feb, High in Jun-Sep
    - park: national parks/reserves → Low in monsoon (closed)
    - ne_monsoon: Northeast nature/hill spots → Low in heavy monsoon (Jun-Sep)
    - beach: beaches → boosted in winter (Nov-Feb), Low in peak monsoon (Jun-Aug)
    - falls: waterfalls → capped at Medium in monsoon
    - landmark: national icons (100k+ reviews) → +20 boost (keeps seasonal shape)
    Returns (adjusted_score, forced_label_or_None).
    """
    otype = OVERRIDE_TYPE.get(place_id) or OVERRIDE_TYPE.get(str(place_id))
    # City-level consistency: if this place isn't individually tagged but its
    # city is a seasonal-closure city, inherit the city's type (fixes Loktak).
    if otype is None:
        info = PLACE_INFO.get(place_id) or PLACE_INFO.get(str(place_id)) or {}
        ccity = info.get('city', '').lower().strip()
        otype = CITY_OVERRIDE.get(ccity)
    if otype == 'winter_closed':
        if month in [12, 1, 2]:  return 5.0, 'Low'
        if month in [6, 7, 8, 9]: return 85.0, 'High'
    if otype == 'park':
        if month in [6, 7, 8, 9]: return 8.0, 'Low'
    if otype == 'ne_monsoon':
        if month in [6, 7, 8, 9]: return 8.0, 'Low'
    if otype == 'beach':
        if month in [11, 12, 1, 2]: return min(100, score + 25), None
        if month in [6, 7, 8]:      return 8.0, 'Low'
    if otype == 'falls':
        if month in [6, 7, 8, 9]: return min(score, 55.0), None
    # landmark boost (only if not overridden by a physical rule above)
    if (place_id in LANDMARK_IDS or str(place_id) in LANDMARK_IDS) and otype not in ('winter_closed','park','ne_monsoon','beach','falls'):
        return min(100, score + 20), None
    return score, None

SEASONS = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Summer',  4: 'Summer', 5: 'Summer',
    6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
    10: 'Post-Monsoon', 11: 'Post-Monsoon'
}

DAY_COL  = {0:'mon_avg',1:'tue_avg',2:'wed_avg',3:'thu_avg',4:'fri_avg',5:'sat_avg',6:'sun_avg'}
DAY_NAME = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

TIME_COL   = {'morning':'morning_avg','afternoon':'afternoon_avg','evening':'evening_avg','night':'night_avg'}
TIME_LABEL = {'morning':'Morning (6am-12pm)','afternoon':'Afternoon (12pm-5pm)','evening':'Evening (5pm-9pm)','night':'Night (9pm-12am)'}
TIME_EMOJI = {'morning':'🌅','afternoon':'☀️','evening':'🌆','night':'🌙'}

REASONS = {
    'High': {
        'Winter'      :['Peak winter tourist season','Pleasant weather attracting visitors','High search interest this month'],
        'Summer'      :['School vacation period','Summer holiday rush','High tourism search interest'],
        'Monsoon'     :['Religious festival activity','Local tourism peak','Seasonal pilgrimage period'],
        'Post-Monsoon':['Post-monsoon pleasant weather','Navratri and Dussehra season','Peak tourist search interest'],
    },
    'Medium': {
        'Winter'      :['Moderate winter season','Weekend tourism activity','Steady seasonal interest'],
        'Summer'      :['School vacation moderate traffic','Summer tourism building up','Moderate search interest'],
        'Monsoon'     :['Monsoon limiting outdoor visits','Indoor and heritage sites popular','Local visitors predominant'],
        'Post-Monsoon':['Early peak season starting','Moderate festival activity','Growing search interest'],
    },
    'Low': {
        'Winter'      :['Off-season for this destination','Visitors prefer other months','Lower seasonal interest'],
        'Summer'      :['Peak heat reduces outdoor visits','Monsoon season approaching','Low search interest this month'],
        'Monsoon'     :['Monsoon reducing tourist footfall','Weather deterring visitors','Limited outdoor access'],
        'Post-Monsoon':['Early season tourists yet to arrive','Pre-festival quiet period','Moderate weather conditions'],
    }
}

TYPE_EMOJI = {
    'Heritage':'🏛️','Religious':'🙏','Beach':'🏖️','Museum':'🏛️',
    'Park':'🌳','Nature':'🌿','Wildlife':'🐯','Hill Station':'⛰️',
    'Market':'🛒','Cave':'🪨','Amusement Park':'🎡','Viewpoint':'👁️','Tourist Spot':'📍',
}

POPULAR_NAMES = ['Taj Mahal','Amber Fort','Goa Beach','Manali','Mysore Palace','Gateway of India','Hawa Mahal','Red Fort']

PUBLIC_HOLIDAYS = {
    (1,26):'Republic Day',(3,25):'Holi',(4,14):'Ambedkar Jayanti',
    (8,15):'Independence Day',(10,2):'Gandhi Jayanti',
    (10,24):'Dussehra',(11,12):'Diwali',(12,25):'Christmas',
}


# ── Helpers ───────────────────────────────────────
def haversine(lat1,lon1,lat2,lon2):
    R=6371
    lat1,lon1,lat2,lon2=map(math.radians,[lat1,lon1,lat2,lon2])
    a=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R*2*math.asin(math.sqrt(max(0,min(1,a))))

def get_city_score(city, month):
    """PRIMARY: validated city-seasonal score (0-100). Returns None if city unknown."""
    if not city:
        return None
    return CITY_SEASON_LC.get((city.lower().strip(), int(month)))

def get_crowd_score(place_id, month):
    """
    Return (score, label) for a place in a month.
    Primary source = city-seasonal score (validated).
    Falls back to legacy place-level label only if city is unknown.
    """
    info = PLACE_INFO.get(place_id) or PLACE_INFO.get(str(place_id)) or {}
    city = info.get('city', '')
    score = get_city_score(city, month)
    if score is not None:
        # apply physical override rules (winter-closed / park / falls / landmark)
        adj_score, forced_label = apply_override(place_id, month, score)
        if forced_label is not None:
            return adj_score, forced_label
        # per-place adjustment: religious places busier in festival-heavy months
        adj_score = min(100, adj_score + religious_festival_boost(place_id, month))
        return adj_score, score_to_label_relative(city, adj_score)
    # fallback: legacy place-level label -> approximate score
    for pid in [place_id, str(place_id)]:
        for m in [month, int(month)]:
            lbl = CROWD_LKP.get((pid, m))
            if lbl:
                approx = {'Low':20,'Medium':50,'High':80}.get(lbl, 50)
                return approx, lbl
    return 50.0, 'Medium'

def get_crowd(place_id, month):
    """Back-compat: label only."""
    return get_crowd_score(place_id, month)[1]

def get_place_coords(place_id):
    idx=PLACE_IDX.get(place_id) or PLACE_IDX.get(str(place_id))
    if idx is None: return 0.0,0.0
    return float(PLACE_LATS[idx]),float(PLACE_LONS[idx])

def get_time_data(place_id):
    return TIME_LKP.get(str(place_id)) or TIME_LKP.get(place_id) or {}

def get_best_months(place_id, city, current_month):
    """Least-crowded months for this city (from the validated seasonal curve)."""
    scores = [(m, get_city_score(city, m)) for m in range(1,13)]
    scores = [(m,s) for m,s in scores if s is not None]
    if not scores:
        return None
    scores.sort(key=lambda x: x[1])
    low_months = [m for m,s in scores[:3]]
    if current_month in low_months:
        return None  # already a quiet month
    names = [calendar.month_abbr[m] for m in low_months]
    return {'months': low_months, 'text': f"Visit in {', '.join(names)} for a quieter experience"}

def compute_datetime_refinement(place_id, date_str, base_score):
    """
    Refine the city-season base score with day-of-week + time-of-day + holiday.
    Returns adjusted (score, confidence, extra_reasons, meta).
    This is a SECONDARY refinement on top of the validated city base.
    """
    dt       = datetime.strptime(date_str,'%Y-%m-%d')
    month    = dt.month
    weekday  = dt.weekday()
    is_wkend = weekday >= 5

    td           = get_time_data(place_id)
    busyness_avg = td.get('busyness_avg',40) or 40
    busyness_max = td.get('busyness_max',busyness_avg) or busyness_avg

    day_col   = DAY_COL[weekday]
    day_avg   = td.get(day_col,busyness_avg) or busyness_avg
    day_ratio = day_avg/busyness_avg if busyness_avg>0 else 1.0

    time_slot = None  # filled by caller
    return dt, month, weekday, is_wkend, day_ratio, td, busyness_max


# ── Routes ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search')
def search():
    q=request.args.get('q','').lower().strip()
    lim=min(int(request.args.get('limit',10)),20)
    if len(q)<2: return jsonify([])
    results=[p for p in SEARCH_INDEX if q in p['key']][:lim]
    return jsonify([{'place_id':r['place_id'],'name':r['name'],'city':r['city'],
                     'state':r['state'],'type':r['type'],'emoji':TYPE_EMOJI.get(r['type'],'📍')}
                    for r in results])

@app.route('/api/popular')
def popular():
    results=[]
    for name in POPULAR_NAMES:
        matches=[p for p in SEARCH_INDEX if name.lower() in p['key']]
        if matches:
            p=matches[0]
            results.append({'place_id':p['place_id'],'name':p['name'],
                            'city':p['city'],'type':p['type'],'emoji':TYPE_EMOJI.get(p['type'],'📍')})
    if len(results)<4:
        results=[{'place_id':p['place_id'],'name':p['name'],'city':p['city'],
                  'type':p['type'],'emoji':TYPE_EMOJI.get(p['type'],'📍')} for p in SEARCH_INDEX[:8]]
    return jsonify(results[:8])

@app.route('/api/predict')
def predict():
    place_id  = request.args.get('place_id','').strip()
    date_str  = request.args.get('date','')
    time_slot = request.args.get('time','afternoon').lower()
    month     = int(request.args.get('month',10))
    year      = int(request.args.get('year',2026))

    if not place_id: return jsonify({'error':'place_id required'}),400
    info=PLACE_INFO.get(place_id) or PLACE_INFO.get(str(place_id))
    if not info: return jsonify({'error':'Place not found'}),404

    city = info.get('city','')
    lat,lon=get_place_coords(place_id)

    # optional date parsing (for date+time refinement)
    weekday=None; is_weekend=None; day_name=None
    time_label=None; time_emoji=None; holiday=None
    if date_str:
        try:
            dt=datetime.strptime(date_str,'%Y-%m-%d')
            month=dt.month; year=dt.year
            weekday=dt.weekday(); is_weekend=weekday>=5
            day_name=DAY_NAME[weekday]
            time_label=TIME_LABEL.get(time_slot); time_emoji=TIME_EMOJI.get(time_slot)
            holiday=PUBLIC_HOLIDAYS.get((month,dt.day))
        except Exception as e:
            print(f'Date parse error: {e}'); date_str=''

    # ── PRIMARY: validated city-seasonal score (+ physical overrides) ──
    base_score, base_label = get_crowd_score(place_id, month)
    score = base_score

    # Is this a hard physical override (winter-closed / park)? If so, the label
    # is a physical fact (snow-locked, park shut) and must NOT be changed by
    # weekend/time nudges. Detect by re-checking the override.
    _otype = OVERRIDE_TYPE.get(place_id) or OVERRIDE_TYPE.get(str(place_id))
    if _otype is None:
        _otype = CITY_OVERRIDE.get(city.lower().strip())
    hard_forced = (_otype == 'winter_closed' and month in [12,1,2,6,7,8,9]) or \
                  (_otype == 'park' and month in [6,7,8,9]) or \
                  (_otype == 'ne_monsoon' and month in [6,7,8,9]) or \
                  (_otype == 'beach' and month in [6,7,8])

    # ── SECONDARY: date/time refinement (small nudge on the city base) ──
    if date_str and not hard_forced:
        td = get_time_data(place_id)
        busyness_avg = td.get('busyness_avg',40) or 40
        busyness_max = td.get('busyness_max',busyness_avg) or busyness_avg
        # weekend nudge
        if is_weekend: score += 6
        else:          score -= 3
        # time-of-day nudge
        tcol = TIME_COL.get(time_slot,'afternoon_avg')
        tavg = td.get(tcol,busyness_avg) or busyness_avg
        tratio = tavg/busyness_max if busyness_max>0 else 0.5
        if   tratio>0.7: score += 5
        elif tratio<0.3: score -= 5
        # holiday nudge
        if holiday: score += 8
        score = max(0, min(100, score))

    # forced physical labels are preserved; otherwise derive from score
    label = base_label if hard_forced else score_to_label_relative(city, score)
    season = SEASONS.get(month,'Winter')

    # confidence: how strongly this month stands out in the city's yearly curve
    year_scores = [get_city_score(city,m) for m in range(1,13)]
    year_scores = [s for s in year_scores if s is not None]
    if year_scores:
        spread = max(year_scores)-min(year_scores)
        conf = int(min(92, 62 + (spread/100)*30))
    else:
        conf = 65

    # reasons
    reasons = list(REASONS.get(label,{}).get(season,['Seasonal crowd pattern']))
    if date_str:
        if is_weekend: reasons.append(f"{day_name} — weekend crowds expected")
        else:          reasons.append(f"{day_name} — typically quieter than weekends")
        if holiday:    reasons.append(f"Public holiday: {holiday}")

    tip = get_best_months(place_id, city, month)

    return jsonify({
        'place_id'  :place_id,'place_name':info.get('place_name',place_id),
        'city'      :city,'state':info.get('state',''),
        'place_type':info.get('place_type',''),'emoji':TYPE_EMOJI.get(info.get('place_type',''),'📍'),
        'date'      :date_str,'month':month,'month_name':calendar.month_name[month],'year':year,
        'score'     :round(score,1),                         # NEW: continuous 0-100 score
        'label'     :label,'base_label':base_label,'label_text':CROWD_LABEL[label],
        'confidence':conf,'color':CROWD_COLOR[label],'icon':CROWD_ICON[label],
        'reasons'   :reasons[:4],'month_tip':tip,
        'day_name'  :day_name,'time_label':time_label,'time_emoji':time_emoji,
        'holiday'   :holiday,'is_weekend':is_weekend,'lat':lat,'lon':lon,
    })

@app.route('/api/recommend')
def recommend():
    place_id=request.args.get('place_id','').strip()
    month   =int(request.args.get('month',10))
    max_dist=float(request.args.get('max_dist',200))
    if not place_id: return jsonify([])
    pid=place_id if place_id in PLACE_IDX else str(place_id)
    if pid not in PLACE_IDX: return jsonify([])

    qinfo=PLACE_INFO.get(pid) or PLACE_INFO.get(str(pid)) or {}
    q_score,_ = get_crowd_score(pid, month)
    idx=PLACE_IDX[pid]; qlat=float(PLACE_LATS[idx]); qlon=float(PLACE_LONS[idx])

    results=[]
    for c in TOP_K.get(pid,[]):
        cid=c['place_id']
        c_score, c_label = get_crowd_score(cid, month)
        dist=haversine(qlat,qlon,c['lat'],c['lon'])
        cinfo=PLACE_INFO.get(cid) or PLACE_INFO.get(str(cid)) or {}
        # recommend only meaningfully-less-crowded nearby places (by continuous score)
        if dist<=max_dist and c_score < q_score - 5:
            results.append({'place_id':cid,'place_name':cinfo.get('place_name',cid),
                'city':cinfo.get('city',''),'state':cinfo.get('state',''),
                'place_type':cinfo.get('place_type',''),'emoji':TYPE_EMOJI.get(cinfo.get('place_type',''),'📍'),
                'score':round(c_score,1),'crowd':c_label,'crowd_text':CROWD_LABEL[c_label],
                'color':CROWD_COLOR[c_label],'distance':round(dist,1),
                'similarity':round(c['similarity'],3),'rating':round(c.get('rating',0.0),1)})
    # sort: lowest score first, then most similar, then nearest
    results.sort(key=lambda x:(x['score'],-x['similarity'],x['distance']))
    return jsonify(results[:5])

@app.route('/api/shap')
def shap():
    path=gdrive_get(DRIVE['shap'],'shap_summary_bar.png')
    return send_file(path,mimetype='image/png')

@app.route('/api/health')
def health():
    return jsonify({'status':'ok','places':len(PLACE_INFO),
                    'city_scores':len(CITY_SEASON),'version':'phase3-cityseason'})

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
