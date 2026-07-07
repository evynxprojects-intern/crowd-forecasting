"""
Crowd Forecast — India Tourism
Phase 2 Flask App — Date + Time Prediction
12,655 tourist places | LightGBM 98.28% | Recommendation Engine
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
    'rec_engine' : '10peKyAKIAGQNzMxSegTtKugUhpuwEy9y',
    'model'      : '1VNfcK_GcdeNz726M8iE-D7sCYXiDqmDK',
    'features'   : '1B7pCpjI1ezrwhjiLqS5ZbX8Sh5_IDa52',
    'shap'       : '1gE68_5TaBhvY4sPw2M_9vYF04wINC9lT',
    'time_lookup': '1ixpJYVEfcoPeqKTDglKBWTpu8XFmUnf-',
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
CROWD_LKP  = ENGINE['crowd_lookup']
REL_LKP    = ENGINE.get('relative_lookup', {})
TOP_K      = ENGINE['top_k_similar']
PLACE_IDX  = ENGINE['place_idx']
PLACE_LATS = ENGINE['place_lats']
PLACE_LONS = ENGINE['place_lons']

time_path = gdrive_get(DRIVE['time_lookup'], 'time_of_day_lookup.json')
with open(time_path) as f:
    TIME_LKP = json.load(f)

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

print(f'Ready: {len(PLACE_INFO):,} places | {len(TIME_LKP):,} time records')

# ── Constants ─────────────────────────────────────
CROWD_RANK  = {'Low': 0, 'Medium': 1, 'High': 2}
CROWD_COLOR = {'High': '#EF4444', 'Medium': '#F97316', 'Low': '#22C55E'}
CROWD_LABEL = {'High': 'Very Busy', 'Medium': 'Moderately Busy', 'Low': 'Less Crowded'}
CROWD_ICON  = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}

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
        'Winter'      :['Off-season for this place type','Visitors prefer other months','Lower seasonal interest'],
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

def get_crowd(place_id,month):
    for pid in [place_id,str(place_id)]:
        for m in [month,int(month)]:
            v=CROWD_LKP.get((pid,m))
            if v: return v
    return 'Medium'

def get_relative(place_id,month):
    for pid in [place_id,str(place_id)]:
        for m in [month,int(month)]:
            v=REL_LKP.get((pid,m))
            if v: return v
    return None

def get_place_coords(place_id):
    idx=PLACE_IDX.get(place_id) or PLACE_IDX.get(str(place_id))
    if idx is None: return 0.0,0.0
    return float(PLACE_LATS[idx]),float(PLACE_LONS[idx])

def get_time_data(place_id):
    return TIME_LKP.get(str(place_id)) or TIME_LKP.get(place_id) or {}

def compute_datetime_prediction(place_id,date_str,time_slot):
    dt       = datetime.strptime(date_str,'%Y-%m-%d')
    month    = dt.month
    weekday  = dt.weekday()
    is_wkend = weekday >= 5

    base_label = get_crowd(place_id,month)
    base_rank  = CROWD_RANK.get(base_label,1)

    td           = get_time_data(place_id)
    busyness_avg = td.get('busyness_avg',40) or 40
    busyness_max = td.get('busyness_max',busyness_avg) or busyness_avg

    day_col   = DAY_COL[weekday]
    day_avg   = td.get(day_col,busyness_avg) or busyness_avg
    day_ratio = day_avg/busyness_avg if busyness_avg>0 else 1.0

    time_col  = TIME_COL.get(time_slot,'afternoon_avg')
    time_avg  = td.get(time_col,busyness_avg) or busyness_avg
    time_ratio= time_avg/busyness_max if busyness_max>0 else 0.5

    holiday_name  = PUBLIC_HOLIDAYS.get((month,dt.day))
    holiday_boost = 0.15 if holiday_name else 0

    same_label = sum(1 for m in range(1,13) if get_crowd(place_id,m)==base_label)
    base_conf  = 0.60+(same_label/12)*0.20

    day_adj  = +0.06 if day_ratio>1.15 else (-0.06 if day_ratio<0.85 else 0.0)
    time_adj = +0.05 if time_ratio>0.7 else (-0.05 if time_ratio<0.3 else 0.0)
    confidence = min(94,int((base_conf+day_adj+time_adj+holiday_boost)*100))

    combined_ratio = day_ratio*(0.5+time_ratio*0.5)
    final_label    = base_label
    if base_label=='Medium':
        if combined_ratio>1.3 or (is_wkend and time_ratio>0.6): final_label='High'
        elif combined_ratio<0.7: final_label='Low'
    elif base_label=='High' and combined_ratio<0.6: final_label='Medium'
    elif base_label=='Low' and combined_ratio>1.4 and is_wkend: final_label='Medium'

    season  = SEASONS.get(month,'Winter')
    reasons = list(REASONS.get(final_label,{}).get(season,['Seasonal crowd pattern']))
    if is_wkend: reasons.append(f"{DAY_NAME[weekday]} — weekend crowds expected")
    else: reasons.append(f"{DAY_NAME[weekday]} — typically quieter than weekends")
    if time_ratio>0.6: reasons.append(f"{TIME_LABEL[time_slot]} is peak time at this place")
    elif time_ratio<0.3: reasons.append(f"{TIME_LABEL[time_slot]} is off-peak at this place")
    if holiday_name: reasons.append(f"Public holiday: {holiday_name}")

    return {
        'label'     :final_label,'base_label':base_label,
        'confidence':confidence,'day_name'  :DAY_NAME[weekday],
        'is_weekend':is_wkend,  'time_slot' :time_slot,
        'time_label':TIME_LABEL[time_slot],'time_emoji':TIME_EMOJI[time_slot],
        'season'    :season,'reasons'  :reasons[:4],'holiday':holiday_name,
    }

def get_month_tip(place_id,current_month):
    low_months=[m for m in range(1,13) if get_relative(place_id,m)=='Low']
    if low_months and current_month not in low_months:
        names=[calendar.month_abbr[m] for m in low_months[:3]]
        return {'months':low_months[:3],'text':f"Visit in {', '.join(names)} for least crowded experience"}
    return None


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

    lat,lon=get_place_coords(place_id)

    if date_str:
        try:
            dt    =datetime.strptime(date_str,'%Y-%m-%d')
            month =dt.month; year=dt.year
            pred  =compute_datetime_prediction(place_id,date_str,time_slot)
            label      =pred['label']; confidence=pred['confidence']
            reasons    =pred['reasons']; day_name  =pred['day_name']
            time_label =pred['time_label']; time_emoji=pred['time_emoji']
            holiday    =pred['holiday']; is_weekend=pred['is_weekend']
            base_label =pred['base_label']
        except Exception as e:
            print(f'Date error: {e}'); date_str=''

    if not date_str:
        label     =get_crowd(place_id,month)
        confidence=min(94,65+sum(1 for m in range(1,13) if get_crowd(place_id,m)==label))
        season    =SEASONS.get(month,'Winter')
        reasons   =list(REASONS.get(label,{}).get(season,['Seasonal crowd pattern']))
        day_name  =None; time_label=None; time_emoji=None
        holiday   =None; is_weekend=None; base_label=label

    tip=get_month_tip(place_id,month)

    return jsonify({
        'place_id'  :place_id,'place_name':info.get('place_name',place_id),
        'city'      :info.get('city',''),'state':info.get('state',''),
        'place_type':info.get('place_type',''),'emoji':TYPE_EMOJI.get(info.get('place_type',''),'📍'),
        'date'      :date_str,'month':month,'month_name':calendar.month_name[month],'year':year,
        'label'     :label,'base_label':base_label,'label_text':CROWD_LABEL[label],
        'confidence':confidence,'color':CROWD_COLOR[label],'icon':CROWD_ICON[label],
        'reasons'   :reasons,'month_tip':tip,
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
    query_crowd=get_crowd(pid,month); query_rank=CROWD_RANK.get(query_crowd,2)
    idx=PLACE_IDX[pid]; qlat=float(PLACE_LATS[idx]); qlon=float(PLACE_LONS[idx])
    results=[]
    for c in TOP_K.get(pid,[]):
        cid=c['place_id']; crowd=get_crowd(cid,month)
        crank=CROWD_RANK.get(crowd,99); dist=haversine(qlat,qlon,c['lat'],c['lon'])
        cinfo=PLACE_INFO.get(cid) or PLACE_INFO.get(str(cid)) or {}
        if dist<=max_dist and crank<query_rank and crowd!='Unknown':
            results.append({'place_id':cid,'place_name':cinfo.get('place_name',cid),
                'city':cinfo.get('city',''),'state':cinfo.get('state',''),
                'place_type':cinfo.get('place_type',''),'emoji':TYPE_EMOJI.get(cinfo.get('place_type',''),'📍'),
                'crowd':crowd,'crowd_text':CROWD_LABEL[crowd],'color':CROWD_COLOR[crowd],
                'distance':round(dist,1),'similarity':round(c['similarity'],3),'rating':round(c.get('rating',0.0),1)})
    results.sort(key=lambda x:(CROWD_RANK.get(x['crowd'],2),-x['similarity'],x['distance']))
    return jsonify(results[:5])

@app.route('/api/shap')
def shap():
    path=gdrive_get(DRIVE['shap'],'shap_summary_bar.png')
    return send_file(path,mimetype='image/png')

@app.route('/api/health')
def health():
    return jsonify({'status':'ok','places':len(PLACE_INFO),'version':'phase2-datetime'})

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
