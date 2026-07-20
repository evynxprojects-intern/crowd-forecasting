"""
Crowd Forecast — India Tourism
Phase 2 Flask App — City-Seasonal Prediction (primary) + Date/Time refinement
12,655 tourist places | City-season model (validated) | Recommendation Engine
"""

import os
import csv
import pickle
import json
import calendar
import math
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template, request, send_file
import gdown

FEEDBACK_FILE = '/tmp/crowd_feedback.csv'

app = Flask(__name__)

# ── Google Drive File IDs ─────────────────────────
DRIVE = {
    'rec_engine'  : '10peKyAKIAGQNzMxSegTtKugUhpuwEy9y',
    'model'       : '1VNfcK_GcdeNz726M8iE-D7sCYXiDqmDK',
    'features'    : '1B7pCpjI1ezrwhjiLqS5ZbX8Sh5_IDa52',
    'shap'        : '1gE68_5TaBhvY4sPw2M_9vYF04wINC9lT',
    'time_lookup' : '1ixpJYVEfcoPeqKTDglKBWTpu8XFmUnf-',
    'city_season' : '1su3VPny5tya6seLPE0DYE2cbYUUynyyk',  # v4 FINAL: physics gates (cold/rain/park from lat+temp+rainfall) + fitted festival weight (w=0.20) + regional festival table — zero name lists
    'overrides'   : '1kSXZPQ0x2wUuGsYd2Qn53u5tg3f97JJ_',    # falls tags + landmarks (incl. 14 govt-verified via ASI data)
    'deviation'   : '1pTJMOR9b2sLZ3XzVgGSeJWz2DYhamupw',    # per-place deviation signal (z_place)
    'reconcile'   : '1eZWye6DdhEpGiayH4LLuiJFSLJZweKh-',    # per-city reconciliation factor
    'dev_config'  : '1UA4Bdn2LhM933v9DmKcpAeuV2Y5EFcva',    # alpha / delta / beta config
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

# Falls tags + landmark IDs (closure/monsoon physics now lives INSIDE the
# v4 city-season scores — derived from latitude/temp/rainfall/type features,
# no name lists; see offline build in Repo 2).
OVERRIDE_TYPE = {}
LANDMARK_IDS  = set()
RELIGIOUS_IDS = set()
try:
    ov_path = gdrive_get(DRIVE['overrides'], 'place_overrides.json')
    with open(ov_path) as f:
        _ov = json.load(f)
    OVERRIDE_TYPE = _ov.get('override_type', {})
    LANDMARK_IDS  = set(_ov.get('landmark_ids', []))
    RELIGIOUS_IDS = set(_ov.get('religious_ids', []))
    print(f'Place tags loaded: {len(OVERRIDE_TYPE)} tagged, {len(LANDMARK_IDS)} landmarks')
except Exception as e:
    print(f'Place tags not loaded (running without): {e}')

# Regional festival table (public calendar facts: Rath Yatra->Puri Jul, etc.)
# Already baked into v4 scores; loaded here so the API can EXPLAIN them.
# Regional festival table — embedded data, versioned with the code.
# month-level entries feed reasons; 'windows' (year-specific, refresh
# yearly for the lunar calendar) drive the festival-DAY boost in /api/predict.
# Regional festival table — embedded data, versioned with the code.
# 'windows' are year-specific (2026-27), verified against panchang where
# marked True; refresh yearly. Edit this list to add festivals — no rebuild.
REGIONAL_FESTIVALS = [{'festival': 'Rath Yatra',
  'month': 7,
  'scope': 'city',
  'value': 'Puri',
  'magnitude': 3,
  'windows': [{'start': '2026-07-16',
               'end': '2026-07-26',
               'label': 'Rath Yatra (Bahuda Jul 24, Suna Besha Jul 25)',
               'verified': True}]},
 {'festival': 'Rath Yatra Ranchi',
  'month': 7,
  'scope': 'city',
  'value': 'Ranchi',
  'magnitude': 2,
  'windows': [{'start': '2026-07-16', 'end': '2026-07-24', 'label': 'Rath Yatra week', 'verified': True}]},
 {'festival': 'Rath Yatra Ahmedabad',
  'month': 7,
  'scope': 'city',
  'value': 'Ahmedabad',
  'magnitude': 2,
  'windows': [{'start': '2026-07-16', 'end': '2026-07-17', 'label': 'Rath Yatra', 'verified': True}]},
 {'festival': 'Pushkar Camel Fair',
  'month': 11,
  'scope': 'city',
  'value': 'Pushkar',
  'magnitude': 3,
  'windows': [{'start': '2026-11-17',
               'end': '2026-11-24',
               'label': 'Pushkar Camel Fair',
               'verified': False}]},
 {'festival': 'Onam',
  'month': 9,
  'scope': 'state',
  'value': 'Kerala',
  'magnitude': 2,
  'windows': [{'start': '2026-08-17',
               'end': '2026-08-27',
               'label': 'Onam (Thiruvonam Aug 26)',
               'verified': True}]},
 {'festival': 'Onam eve', 'month': 8, 'scope': 'state', 'value': 'Kerala', 'magnitude': 1},
 {'festival': 'Pongal',
  'month': 1,
  'scope': 'state',
  'value': 'Tamil Nadu',
  'magnitude': 2,
  'windows': [{'start': '2026-01-14', 'end': '2026-01-17', 'label': 'Pongal', 'verified': True}]},
 {'festival': 'Durga Puja',
  'month': 10,
  'scope': 'city',
  'value': 'Kolkata',
  'magnitude': 3,
  'windows': [{'start': '2026-10-15',
               'end': '2026-10-20',
               'label': 'Durga Puja (Dashami Oct 20)',
               'verified': True}]},
 {'festival': 'Durga Puja WB',
  'month': 10,
  'scope': 'state',
  'value': 'West Bengal',
  'magnitude': 2,
  'windows': [{'start': '2026-10-15',
               'end': '2026-10-20',
               'label': 'Durga Puja (Dashami Oct 20)',
               'verified': True}]},
 {'festival': 'Ganesh Chaturthi',
  'month': 9,
  'scope': 'city',
  'value': 'Mumbai',
  'magnitude': 3,
  'windows': [{'start': '2026-09-14',
               'end': '2026-09-24',
               'label': 'Ganesh festival (visarjan ~Sep 24)',
               'verified': True}]},
 {'festival': 'Ganesh Pune',
  'month': 9,
  'scope': 'city',
  'value': 'Pune',
  'magnitude': 2,
  'windows': [{'start': '2026-09-14',
               'end': '2026-09-24',
               'label': 'Ganesh festival (visarjan ~Sep 24)',
               'verified': True}]},
 {'festival': 'Hornbill Festival',
  'month': 12,
  'scope': 'city',
  'value': 'Kohima',
  'magnitude': 3,
  'windows': [{'start': '2026-12-01', 'end': '2026-12-10', 'label': 'Hornbill Festival', 'verified': True}]},
 {'festival': 'Bihu',
  'month': 4,
  'scope': 'state',
  'value': 'Assam',
  'magnitude': 2,
  'windows': [{'start': '2026-04-14', 'end': '2026-04-16', 'label': 'Rongali Bihu', 'verified': True}]},
 {'festival': 'Baisakhi',
  'month': 4,
  'scope': 'city',
  'value': 'Amritsar',
  'magnitude': 2,
  'windows': [{'start': '2026-04-13', 'end': '2026-04-14', 'label': 'Baisakhi', 'verified': True}]},
 {'festival': 'Navratri Garba',
  'month': 10,
  'scope': 'city',
  'value': 'Ahmedabad',
  'magnitude': 3,
  'windows': [{'start': '2026-10-11',
               'end': '2026-10-20',
               'label': 'Navratri Garba (Oct 11-20)',
               'verified': True}]},
 {'festival': 'Navratri Gujarat',
  'month': 10,
  'scope': 'state',
  'value': 'Gujarat',
  'magnitude': 2,
  'windows': [{'start': '2026-10-11',
               'end': '2026-10-20',
               'label': 'Navratri (Oct 11-20)',
               'verified': True}]},
 {'festival': 'Mysore Dasara',
  'month': 10,
  'scope': 'city',
  'value': 'Mysore',
  'magnitude': 3,
  'windows': [{'start': '2026-10-11',
               'end': '2026-10-20',
               'label': 'Mysore Dasara (Oct 11-20)',
               'verified': True}]},
 {'festival': 'Hemis Festival', 'month': 7, 'scope': 'city', 'value': 'Leh', 'magnitude': 2},
 {'festival': 'Desert Festival', 'month': 2, 'scope': 'city', 'value': 'Jaisalmer', 'magnitude': 2},
 {'festival': 'Rann Utsav Dec',
  'month': 12,
  'scope': 'city',
  'value': 'Kutch',
  'magnitude': 3,
  'windows': [{'start': '2026-11-01', 'end': '2027-02-28', 'label': 'Rann Utsav season', 'verified': True}]},
 {'festival': 'Rann Utsav Jan', 'month': 1, 'scope': 'city', 'value': 'Kutch', 'magnitude': 3},
 {'festival': 'Konark Dance Fest',
  'month': 12,
  'scope': 'city',
  'value': 'Konark',
  'magnitude': 2,
  'windows': [{'start': '2026-12-01',
               'end': '2026-12-05',
               'label': 'Konark Dance Festival',
               'verified': True}]},
 {'festival': 'Khajuraho Dance Fest',
  'month': 2,
  'scope': 'city',
  'value': 'Khajuraho',
  'magnitude': 2,
  'windows': [{'start': '2026-02-20',
               'end': '2026-02-26',
               'label': 'Khajuraho Dance Festival',
               'verified': True}]},
 {'festival': 'Thrissur Pooram',
  'month': 5,
  'scope': 'city',
  'value': 'Thrissur',
  'magnitude': 3,
  'windows': [{'start': '2026-04-26',
               'end': '2026-04-27',
               'label': 'Thrissur Pooram (approx)',
               'verified': False}]},
 {'festival': 'Chhath Puja',
  'month': 11,
  'scope': 'city',
  'value': 'Patna',
  'magnitude': 3,
  'windows': [{'start': '2026-11-13',
               'end': '2026-11-16',
               'label': 'Chhath Puja (main day Nov 15)',
               'verified': True}]},
 {'festival': 'Chhath Varanasi',
  'month': 11,
  'scope': 'city',
  'value': 'Varanasi',
  'magnitude': 2,
  'windows': [{'start': '2026-11-13',
               'end': '2026-11-16',
               'label': 'Chhath Puja (main day Nov 15)',
               'verified': True}]},
 {'festival': 'Goa Carnival',
  'month': 2,
  'scope': 'city',
  'value': 'Goa',
  'magnitude': 2,
  'windows': [{'start': '2026-02-14', 'end': '2026-02-17', 'label': 'Goa Carnival', 'verified': True}]},
 {'festival': 'NYE Goa',
  'month': 12,
  'scope': 'city',
  'value': 'Goa',
  'magnitude': 2,
  'windows': [{'start': '2026-12-24',
               'end': '2027-01-01',
               'label': 'Christmas–New Year peak',
               'verified': True}]},
 {'festival': 'Amarnath Yatra',
  'month': 7,
  'scope': 'city',
  'value': 'Pahalgam',
  'magnitude': 3,
  'windows': [{'start': '2026-07-01',
               'end': '2026-08-09',
               'label': 'Amarnath Yatra season (approx)',
               'verified': False}]},
 {'festival': 'Amarnath Aug', 'month': 8, 'scope': 'city', 'value': 'Pahalgam', 'magnitude': 2},
 {'festival': 'Teej', 'month': 8, 'scope': 'city', 'value': 'Jaipur', 'magnitude': 1},
 {'festival': 'Gangaur', 'month': 3, 'scope': 'city', 'value': 'Jaipur', 'magnitude': 1},
 {'festival': 'Holi Mathura',
  'month': 3,
  'scope': 'city',
  'value': 'Mathura',
  'magnitude': 3,
  'windows': [{'start': '2026-02-25',
               'end': '2026-03-04',
               'label': 'Braj Holi week (approx)',
               'verified': False}]},
 {'festival': 'Holi Vrindavan',
  'month': 3,
  'scope': 'city',
  'value': 'Vrindavan',
  'magnitude': 3,
  'windows': [{'start': '2026-02-25',
               'end': '2026-03-04',
               'label': 'Braj Holi week (approx)',
               'verified': False}]},
 {'festival': 'Buddha Purnima', 'month': 5, 'scope': 'city', 'value': 'Bodh Gaya', 'magnitude': 2},
 {'festival': 'Buddha Purnima Sarnath', 'month': 5, 'scope': 'city', 'value': 'Sarnath', 'magnitude': 1},
 {'festival': 'Magh Mela',
  'month': 1,
  'scope': 'city',
  'value': 'Prayagraj',
  'magnitude': 2,
  'windows': [{'start': '2026-01-03',
               'end': '2026-02-17',
               'label': 'Magh Mela (approx)',
               'verified': False}]},
 {'festival': 'Intl Yoga Festival',
  'month': 3,
  'scope': 'city',
  'value': 'Rishikesh',
  'magnitude': 2,
  'windows': [{'start': '2026-03-01',
               'end': '2026-03-07',
               'label': 'International Yoga Festival',
               'verified': True}]},
 {'festival': 'Tawang Festival', 'month': 10, 'scope': 'city', 'value': 'Tawang', 'magnitude': 2},
 {'festival': 'Shimla Summer Fest', 'month': 6, 'scope': 'city', 'value': 'Shimla', 'magnitude': 1},
 {'festival': 'Kanwar Yatra',
  'month': 8,
  'scope': 'city',
  'value': 'Haridwar',
  'magnitude': 3,
  'windows': [{'start': '2026-07-30',
               'end': '2026-08-12',
               'label': 'Kanwar Yatra (Shravan)',
               'verified': False}]},
 {'festival': 'Kanwar Yatra Rishikesh',
  'month': 8,
  'scope': 'city',
  'value': 'Rishikesh',
  'magnitude': 2,
  'windows': [{'start': '2026-07-30',
               'end': '2026-08-12',
               'label': 'Kanwar Yatra (Shravan)',
               'verified': False}]},
 {'festival': 'Janmashtami Mathura',
  'month': 9,
  'scope': 'city',
  'value': 'Mathura',
  'magnitude': 3,
  'windows': [{'start': '2026-09-03',
               'end': '2026-09-05',
               'label': 'Janmashtami (Sep 4)',
               'verified': True}]},
 {'festival': 'Janmashtami Vrindavan',
  'month': 9,
  'scope': 'city',
  'value': 'Vrindavan',
  'magnitude': 3,
  'windows': [{'start': '2026-09-03',
               'end': '2026-09-05',
               'label': 'Janmashtami (Sep 4)',
               'verified': True}]},
 {'festival': 'Independence Day Delhi',
  'month': 8,
  'scope': 'city',
  'value': 'Delhi',
  'magnitude': 1,
  'windows': [{'start': '2026-08-15',
               'end': '2026-08-15',
               'label': 'Independence Day (Red Fort)',
               'verified': True}]},
 {'festival': 'Nehru Trophy Boat Race',
  'month': 8,
  'scope': 'city',
  'value': 'Alleppey',
  'magnitude': 2,
  'windows': [{'start': '2026-08-08',
               'end': '2026-08-08',
               'label': 'Nehru Trophy Boat Race (approx)',
               'verified': False}]},
 {'festival': 'Kullu Dussehra',
  'month': 10,
  'scope': 'city',
  'value': 'Kullu',
  'magnitude': 3,
  'windows': [{'start': '2026-10-20',
               'end': '2026-10-26',
               'label': 'Kullu Dussehra (starts Oct 20)',
               'verified': True}]},
 {'festival': 'Dev Deepawali',
  'month': 11,
  'scope': 'city',
  'value': 'Varanasi',
  'magnitude': 3,
  'windows': [{'start': '2026-11-23',
               'end': '2026-11-24',
               'label': 'Dev Deepawali (Nov 24)',
               'verified': True}]},
 {'festival': 'Diwali Amritsar',
  'month': 11,
  'scope': 'city',
  'value': 'Amritsar',
  'magnitude': 2,
  'windows': [{'start': '2026-11-07',
               'end': '2026-11-09',
               'label': 'Diwali / Bandi Chhor Divas (Nov 8)',
               'verified': True}]},
 {'festival': 'Guru Nanak Jayanti',
  'month': 11,
  'scope': 'city',
  'value': 'Amritsar',
  'magnitude': 2,
  'windows': [{'start': '2026-11-23',
               'end': '2026-11-24',
               'label': 'Guru Nanak Jayanti (Nov 24)',
               'verified': True}]},
 {'festival': 'Sonepur Mela',
  'month': 11,
  'scope': 'city',
  'value': 'Sonepur',
  'magnitude': 2,
  'windows': [{'start': '2026-11-24',
               'end': '2026-12-20',
               'label': 'Sonepur Mela (approx)',
               'verified': False}]},
 {'festival': 'Margazhi Season',
  'month': 12,
  'scope': 'city',
  'value': 'Chennai',
  'magnitude': 1,
  'windows': [{'start': '2026-12-15',
               'end': '2027-01-15',
               'label': 'Margazhi music season',
               'verified': True}]},
 {'festival': 'Mount Abu Winter Fest',
  'month': 12,
  'scope': 'city',
  'value': 'Mount Abu',
  'magnitude': 1,
  'windows': [{'start': '2026-12-29', 'end': '2026-12-31', 'label': 'Winter Festival', 'verified': False}]},
 {'festival': 'Uttarayan Kite Festival',
  'month': 1,
  'scope': 'city',
  'value': 'Ahmedabad',
  'magnitude': 2,
  'windows': [{'start': '2027-01-13',
               'end': '2027-01-15',
               'label': 'Uttarayan Kite Festival',
               'verified': True}]},
 {'festival': 'Ganga Sagar Mela',
  'month': 1,
  'scope': 'city',
  'value': 'Kolkata',
  'magnitude': 1,
  'windows': [{'start': '2027-01-12',
               'end': '2027-01-15',
               'label': 'Ganga Sagar Mela (approx)',
               'verified': False}]},
 {'festival': 'Republic Day Delhi',
  'month': 1,
  'scope': 'city',
  'value': 'Delhi',
  'magnitude': 1,
  'windows': [{'start': '2027-01-26', 'end': '2027-01-26', 'label': 'Republic Day', 'verified': True}]}]
print(f'Regional festivals: {len(REGIONAL_FESTIVALS)} entries (embedded)')

def festivals_for(city, state, month):
    """Return festival names active in this city/state for this month."""
    out = []
    cl, sl = (city or '').lower(), (state or '').lower()
    for f in REGIONAL_FESTIVALS:
        try:
            if int(f.get('month', 0)) != int(month):
                continue
            val = str(f.get('value', '')).lower()
            if f.get('scope') == 'city' and val and val in cl:
                out.append(f['festival'])
            elif f.get('scope') == 'state' and val and val == sl:
                out.append(f['festival'])
        except Exception:
            continue
    return out

def festival_day_for(city, state, date_iso):
    """If this exact date falls inside a festival window for this city/state,
    return the window label (e.g. 'Rath Yatra (Bahuda Jul 24...)') else None.
    Windows are year-specific data in regional_festivals.json — refresh yearly."""
    cl, sl = (city or '').lower(), (state or '').lower()
    for f in REGIONAL_FESTIVALS:
        val = str(f.get('value', '')).lower()
        matches = (f.get('scope') == 'city' and val and val in cl) or \
                  (f.get('scope') == 'state' and val and val == sl)
        if not matches:
            continue
        for w in f.get('windows', []):
            try:
                if w['start'] <= date_iso <= w['end']:
                    return w.get('label', f.get('festival', 'Festival'))
            except Exception:
                continue
    return None

# ── Phase 2: place-level deviation framework (bounded residual + reconciliation) ──
PLACE_DEVIATION = {}
CITY_RECONCILE  = {}
DEV_ALPHA, DEV_DELTA = 0.15, 0.25
try:
    dev_path = gdrive_get(DRIVE['deviation'], 'place_deviation.json')
    with open(dev_path) as f:
        PLACE_DEVIATION = json.load(f)
    rec_path = gdrive_get(DRIVE['reconcile'], 'city_reconcile.json')
    with open(rec_path) as f:
        CITY_RECONCILE = json.load(f)
    cfg_path = gdrive_get(DRIVE['dev_config'], 'deviation_config.json')
    with open(cfg_path) as f:
        _cfg = json.load(f)
    DEV_ALPHA = _cfg.get('alpha', 0.15)
    DEV_DELTA = _cfg.get('delta', 0.25)
    print(f'Deviation framework: {len(PLACE_DEVIATION)} places, {len(CITY_RECONCILE)} cities, alpha={DEV_ALPHA}')
except Exception as e:
    print(f'Deviation framework not loaded (places default to city score): {e}')

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

# ── Place-level adjustments (minimal — seasonal physics lives in v4 scores) ──
def apply_override(place_id, month, score):
    """
    v4 scores already encode all seasonal physics per city (cold gate from
    latitude+annual temp, rain gate from annual rainfall, park monsoon closure
    from type flags, regional festivals from the festival table) — derived
    from features, zero name lists. Only two place-level adjustments remain:
    - falls: waterfalls capped at Medium in monsoon (scenic but not packed)
    - landmark: 100k+ reviews OR 250k+ govt-counted visitors -> +20 boost
    Returns (adjusted_score, forced_label_or_None). forced_label is always
    None now; labels derive from the score via relative ranking.
    """
    otype = OVERRIDE_TYPE.get(place_id) or OVERRIDE_TYPE.get(str(place_id))
    if otype == 'falls' and month in [6, 7, 8, 9]:
        return min(score, 55.0), None
    if place_id in LANDMARK_IDS or str(place_id) in LANDMARK_IDS:
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

def apply_place_deviation(place_id, city, city_score):
    """
    Phase 2: bounded residual + reconciliation.
    Adjust the city baseline for this specific place using its precomputed
    deviation signal (popularity/busyness/velocity/rating, standardized within
    the city). Bounded so places stay close to the city, and reconciled so the
    city average is preserved.
    Returns (adjusted_score, deviation_pct) where deviation_pct is for explanation.
    """
    z = PLACE_DEVIATION.get(str(place_id)) or PLACE_DEVIATION.get(place_id)
    if z is None:
        return city_score, 0.0   # no data -> defaults exactly to city score
    # bounded multiplicative offset
    adj = max(-DEV_DELTA, min(DEV_DELTA, DEV_ALPHA * z))
    raw = city_score * (1 + adj)
    # reconciliation: preserve city average
    rf = CITY_RECONCILE.get(city) or CITY_RECONCILE.get(city.strip()) or 1.0
    final = max(0, min(100, raw * rf))
    return final, round((final - city_score) / city_score * 100, 1) if city_score else 0.0

def get_crowd_score(place_id, month):
    """
    Return (score, label) for a place in a month.
    Layer 1: validated city-seasonal score (baseline).
    Layer 2: physical override rules (winter-closed / park / beach / landmark).
    Layer 3: bounded place-level deviation (popularity/type), reconciled to city.
    """
    info = PLACE_INFO.get(place_id) or PLACE_INFO.get(str(place_id)) or {}
    city = info.get('city', '')
    score = get_city_score(city, month)
    if score is not None:
        # Layer 2: physical overrides (hard closures return a forced label)
        adj_score, forced_label = apply_override(place_id, month, score)
        if forced_label is not None:
            return adj_score, forced_label
        # Layer 3: bounded place-level deviation on top of the city base
        adj_score, _ = apply_place_deviation(place_id, city, adj_score)
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

    # ── PRIMARY: v4 city-seasonal score (physics + festivals baked in) ──
    base_score, base_label = get_crowd_score(place_id, month)
    score = base_score

    # ── SECONDARY: date/time refinement (small nudge on the city base) ──
    # Skipped when the seasonal score indicates near-closure (physically shut
    # places shouldn't be "un-closed" by a weekend bump).
    if date_str and base_score >= 12:
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

    # ── FESTIVAL DAY: exact date falls in a festival window for this city ──
    # (dates are data in regional_festivals.json, refreshed yearly for the
    # lunar calendar; e.g. Rath Yatra 2026 = Jul 16-26, Bahuda = Jul 24)
    fest_day = festival_day_for(city, info.get('state',''), date_str) if date_str else None
    if fest_day:
        score = min(100, max(score + 35, 80))

    # label derives from the score (relative to the city's own yearly range)
    label = score_to_label_relative(city, score)
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
    if fest_day:
        reasons = [f"🎉 {fest_day} — expect very heavy crowds"] + reasons
    else:
        _fests = festivals_for(city, info.get('state',''), month)
        for _f in _fests[:2]:
            reasons.insert(0, f"🎉 {_f} this month — expect festive crowds")
    if date_str:
        if is_weekend: reasons.append(f"{day_name} — weekend crowds expected")
        elif not fest_day: reasons.append(f"{day_name} — typically quieter than weekends")
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

@app.route('/api/festivals')
def api_festivals():
    """Regional festivals — all, or filtered by ?city= and/or ?month=."""
    city  = request.args.get('city', '').strip()
    month = request.args.get('month', type=int)
    out = REGIONAL_FESTIVALS
    if month:
        out = [f for f in out if int(f.get('month', 0)) == month]
    if city:
        cl = city.lower()
        out = [f for f in out if (f.get('scope') == 'city' and str(f.get('value','')).lower() in cl)
               or (f.get('scope') == 'state')]
    return jsonify({'count': len(out), 'festivals': out})


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Record a user's real-world crowd observation — the path to real ground truth."""
    data = request.get_json(silent=True) or {}
    place_id = str(data.get('place_id', '')).strip()
    user_label = str(data.get('user_label', '')).strip()
    if not place_id or user_label not in ('Low', 'Medium', 'High'):
        return jsonify({'error': 'place_id and user_label (Low/Medium/High) required'}), 400
    info = PLACE_INFO.get(place_id) or PLACE_INFO.get(str(place_id)) or {}
    row = [datetime.now(timezone.utc).isoformat(), place_id,
           info.get('place_name', ''), info.get('city', ''),
           data.get('month', ''), data.get('predicted_score', ''),
           data.get('predicted_label', ''), user_label]
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['ts', 'place_id', 'place_name', 'city', 'month',
                        'predicted_score', 'predicted_label', 'user_label'])
        w.writerow(row)
    return jsonify({'status': 'recorded', 'thanks': True})


@app.route('/api/feedback/export')
def export_feedback():
    """Download all collected feedback as CSV (for calibration/retraining).
    NOTE: /tmp is wiped on Render restarts — export regularly."""
    if not os.path.exists(FEEDBACK_FILE):
        return jsonify({'rows': 0, 'message': 'no feedback yet'})
    return send_file(FEEDBACK_FILE, mimetype='text/csv',
                     as_attachment=True, download_name='crowd_feedback.csv')


@app.route('/api/health')
def health():
    return jsonify({'status':'ok','places':len(PLACE_INFO),
                    'city_scores':len(CITY_SEASON),'version':'phase3-cityseason'})

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
