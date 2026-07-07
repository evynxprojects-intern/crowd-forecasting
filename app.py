"""
Crowd Forecast — India Tourism
Phase 2 Flask App
12,655 tourist places | LightGBM 98.28% | Recommendation Engine
"""

import os
import pickle
import json
import calendar
import math
from flask import Flask, jsonify, render_template, request, send_file
import gdown

app = Flask(__name__)

# ── Google Drive File IDs ─────────────────────────
DRIVE = {
    'rec_engine': '10peKyAKIAGQNzMxSegTtKugUhpuwEy9y',
    'model'     : '1VNfcK_GcdeNz726M8iE-D7sCYXiDqmDK',
    'features'  : '1B7pCpjI1ezrwhjiLqS5ZbX8Sh5_IDa52',
    'shap'      : '1gE68_5TaBhvY4sPw2M_9vYF04wINC9lT',
}

CACHE = '/tmp/crowd_cache'
os.makedirs(CACHE, exist_ok=True)


def gdrive_get(file_id, filename):
    """Download from Google Drive if not cached."""
    path = os.path.join(CACHE, filename)
    if not os.path.exists(path):
        print(f'⬇️  Downloading {filename}...')
        gdown.download(
            f'https://drive.google.com/uc?id={file_id}',
            path, quiet=False)
        print(f'✅ {filename} ({os.path.getsize(path)/1e6:.1f} MB)')
    else:
        print(f'✅ {filename} cached')
    return path


# ── Load at startup ───────────────────────────────
print('🚀 Loading Crowd Forecast engine...')
rec_path = gdrive_get(DRIVE['rec_engine'], 'recommendation_engine_final.pkl')

with open(rec_path, 'rb') as f:
    ENGINE = pickle.load(f)

PLACE_INFO   = ENGINE['place_info_lookup']
CROWD_LKP    = ENGINE['crowd_lookup']
REL_LKP      = ENGINE.get('relative_lookup', {})
TOP_K        = ENGINE['top_k_similar']
PLACE_IDX    = ENGINE['place_idx']
PLACE_LATS   = ENGINE['place_lats']
PLACE_LONS   = ENGINE['place_lons']
PLACE_IDS    = ENGINE['place_ids']

# Build fast search index
SEARCH_INDEX = sorted([
    {
        'place_id' : pid,
        'name'     : info.get('place_name', ''),
        'city'     : info.get('city', ''),
        'state'    : info.get('state', ''),
        'type'     : info.get('place_type', ''),
        'key'      : f"{info.get('place_name','')} {info.get('city','')} {info.get('state','')}".lower()
    }
    for pid, info in PLACE_INFO.items()
], key=lambda x: x['name'])

print(f'✅ Engine ready | {len(PLACE_INFO):,} places | {len(CROWD_LKP):,} predictions')


# ── Constants ─────────────────────────────────────
CROWD_RANK   = {'Low': 0, 'Medium': 1, 'High': 2}
CROWD_COLOR  = {'High': '#EF4444', 'Medium': '#F97316', 'Low': '#22C55E'}
CROWD_LABEL  = {'High': 'Very Busy', 'Medium': 'Moderately Busy', 'Low': 'Less Crowded'}
CROWD_ICON   = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}

SEASONS = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Summer',  4: 'Summer', 5: 'Summer',
    6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
    10: 'Post-Monsoon', 11: 'Post-Monsoon'
}

REASONS = {
    'High': {
        'Winter'       : ['Peak winter tourist season', 'Pleasant weather attracting visitors', 'High search interest this month'],
        'Summer'       : ['School vacation period', 'Summer holiday rush', 'High tourism search interest'],
        'Monsoon'      : ['Religious festival activity', 'Local tourism peak', 'Seasonal pilgrimage period'],
        'Post-Monsoon' : ['Post-monsoon pleasant weather', 'Navratri and Dussehra season', 'Peak tourist search interest'],
    },
    'Medium': {
        'Winter'       : ['Moderate winter season', 'Weekend tourism activity', 'Steady seasonal interest'],
        'Summer'       : ['School vacation moderate traffic', 'Summer tourism building up', 'Moderate search interest'],
        'Monsoon'      : ['Monsoon limiting outdoor visits', 'Indoor and heritage sites popular', 'Local visitors predominant'],
        'Post-Monsoon' : ['Early peak season starting', 'Moderate festival activity', 'Growing search interest'],
    },
    'Low': {
        'Winter'       : ['Off-season for this place type', 'Visitors prefer other months', 'Lower seasonal interest'],
        'Summer'       : ['Peak heat reduces outdoor visits', 'Monsoon season approaching', 'Low search interest this month'],
        'Monsoon'      : ['Monsoon reducing tourist footfall', 'Weather deterring visitors', 'Limited outdoor access'],
        'Post-Monsoon' : ['Early season — tourists yet to arrive', 'Pre-festival quiet period', 'Moderate weather conditions'],
    }
}

TYPE_EMOJI = {
    'Heritage'      : '🏛️',
    'Religious'     : '🙏',
    'Beach'         : '🏖️',
    'Museum'        : '🏛️',
    'Park'          : '🌳',
    'Nature'        : '🌿',
    'Wildlife'      : '🐯',
    'Hill Station'  : '⛰️',
    'Market'        : '🛒',
    'Cave'          : '🪨',
    'Amusement Park': '🎡',
    'Viewpoint'     : '👁️',
    'Tourist Spot'  : '📍',
}

POPULAR_NAMES = [
    'Taj Mahal', 'Amber Fort', 'Goa Beach',
    'Manali', 'Mysore Palace', 'Gateway of India',
    'Hawa Mahal', 'Red Fort'
]


# ── Helpers ───────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + \
        math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, min(1, a))))


def get_crowd(place_id, month):
    """Get crowd label — try multiple key formats."""
    for pid in [place_id, str(place_id)]:
        for m in [month, int(month)]:
            v = CROWD_LKP.get((pid, m))
            if v:
                return v
    return 'Medium'


def get_relative(place_id, month):
    """Get relative crowd label (within-place)."""
    for pid in [place_id, str(place_id)]:
        for m in [month, int(month)]:
            v = REL_LKP.get((pid, m))
            if v:
                return v
    return None


def get_confidence(place_id, label):
    """Compute confidence from label consistency across 12 months."""
    same = sum(1 for m in range(1, 13)
               if get_crowd(place_id, m) == label)
    base = {'High': 72, 'Medium': 63, 'Low': 70}[label]
    return min(94, int(base + (same / 12) * 18))


def get_month_tip(place_id, current_month):
    """Suggest best low-crowd months for this place."""
    low_months = [m for m in range(1, 13)
                  if get_relative(place_id, m) == 'Low']
    if low_months and current_month not in low_months:
        names = [calendar.month_abbr[m] for m in low_months[:3]]
        return {
            'months': low_months[:3],
            'text'  : f"Visit in {', '.join(names)} for least crowded experience"
        }
    return None


def get_place_coords(place_id):
    idx = PLACE_IDX.get(place_id) or PLACE_IDX.get(str(place_id))
    if idx is None:
        return 0.0, 0.0
    return float(PLACE_LATS[idx]), float(PLACE_LONS[idx])


# ── Routes ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/search')
def search():
    q   = request.args.get('q', '').lower().strip()
    lim = min(int(request.args.get('limit', 10)), 20)

    if len(q) < 2:
        return jsonify([])

    results = [p for p in SEARCH_INDEX if q in p['key']][:lim]

    return jsonify([{
        'place_id': r['place_id'],
        'name'    : r['name'],
        'city'    : r['city'],
        'state'   : r['state'],
        'type'    : r['type'],
        'emoji'   : TYPE_EMOJI.get(r['type'], '📍'),
    } for r in results])


@app.route('/api/popular')
def popular():
    results = []
    for name in POPULAR_NAMES:
        matches = [p for p in SEARCH_INDEX if name.lower() in p['key']]
        if matches:
            p = matches[0]
            results.append({
                'place_id': p['place_id'],
                'name'    : p['name'],
                'city'    : p['city'],
                'type'    : p['type'],
                'emoji'   : TYPE_EMOJI.get(p['type'], '📍'),
            })

    if len(results) < 4:
        results = [{
            'place_id': p['place_id'],
            'name'    : p['name'],
            'city'    : p['city'],
            'type'    : p['type'],
            'emoji'   : TYPE_EMOJI.get(p['type'], '📍'),
        } for p in SEARCH_INDEX[:8]]

    return jsonify(results[:8])


@app.route('/api/predict')
def predict():
    place_id = request.args.get('place_id', '').strip()
    month    = int(request.args.get('month', 10))
    year     = int(request.args.get('year', 2026))

    if not place_id:
        return jsonify({'error': 'place_id required'}), 400

    info = PLACE_INFO.get(place_id) or PLACE_INFO.get(str(place_id))
    if not info:
        return jsonify({'error': 'Place not found'}), 404

    label      = get_crowd(place_id, month)
    confidence = get_confidence(place_id, label)
    season     = SEASONS.get(month, 'Winter')
    reasons    = REASONS.get(label, {}).get(season, ['Seasonal crowd pattern'])
    tip        = get_month_tip(place_id, month)
    lat, lon   = get_place_coords(place_id)

    return jsonify({
        'place_id'   : place_id,
        'place_name' : info.get('place_name', place_id),
        'city'       : info.get('city', ''),
        'state'      : info.get('state', ''),
        'place_type' : info.get('place_type', ''),
        'emoji'      : TYPE_EMOJI.get(info.get('place_type', ''), '📍'),
        'month'      : month,
        'month_name' : calendar.month_name[month],
        'year'       : year,
        'label'      : label,
        'label_text' : CROWD_LABEL[label],
        'confidence' : confidence,
        'color'      : CROWD_COLOR[label],
        'icon'       : CROWD_ICON[label],
        'reasons'    : reasons,
        'month_tip'  : tip,
        'lat'        : lat,
        'lon'        : lon,
        'season'     : season,
    })


@app.route('/api/recommend')
def recommend():
    place_id = request.args.get('place_id', '').strip()
    month    = int(request.args.get('month', 10))
    max_dist = float(request.args.get('max_dist', 200))

    if not place_id:
        return jsonify([])

    pid = place_id if place_id in PLACE_IDX else str(place_id)
    if pid not in PLACE_IDX:
        return jsonify([])

    query_crowd = get_crowd(pid, month)
    query_rank  = CROWD_RANK.get(query_crowd, 2)
    qlat, qlon  = get_place_coords(pid)

    results = []
    for c in TOP_K.get(pid, []):
        cid   = c['place_id']
        crowd = get_crowd(cid, month)
        crank = CROWD_RANK.get(crowd, 99)
        dist  = haversine(qlat, qlon, c['lat'], c['lon'])
        cinfo = PLACE_INFO.get(cid) or PLACE_INFO.get(str(cid)) or {}

        if dist <= max_dist and crank < query_rank and crowd != 'Unknown':
            results.append({
                'place_id'  : cid,
                'place_name': cinfo.get('place_name', cid),
                'city'      : cinfo.get('city', ''),
                'state'     : cinfo.get('state', ''),
                'place_type': cinfo.get('place_type', ''),
                'emoji'     : TYPE_EMOJI.get(cinfo.get('place_type', ''), '📍'),
                'crowd'     : crowd,
                'crowd_text': CROWD_LABEL[crowd],
                'color'     : CROWD_COLOR[crowd],
                'distance'  : round(dist, 1),
                'similarity': round(c['similarity'], 3),
                'rating'    : round(c.get('rating', 0.0), 1),
            })

    results.sort(key=lambda x: (
        CROWD_RANK.get(x['crowd'], 2),
        -x['similarity'],
         x['distance']
    ))
    return jsonify(results[:5])


@app.route('/api/shap')
def shap():
    path = gdrive_get(DRIVE['shap'], 'shap_summary_bar.png')
    return send_file(path, mimetype='image/png')


@app.route('/api/health')
def health():
    return jsonify({
        'status' : 'ok',
        'places' : len(PLACE_INFO),
        'version': 'phase2'
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)))
