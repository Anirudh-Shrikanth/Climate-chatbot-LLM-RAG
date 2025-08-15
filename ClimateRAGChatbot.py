# Core dependencies
import os
import json
import gzip
import shutil
import subprocess
import re
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP and ML
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Geocoding
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# APIs
import requests
import openai
from datasets import load_dataset

# Meteostat
import meteostat
from meteostat import Point, Daily

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

@dataclass
class ClimateDocument:
    """Structured climate data document"""
    content: str
    source: str
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    doc_id: Optional[str] = None

    def __post_init__(self):
      if self.doc_id is None:
        import hashlib, json
        stable_fingerprint = json.dumps(
            {
                "source": self.source,
                "doc_type": self.doc_type,
                "date": self.metadata.get("date"),           # e.g., '2023-07-15'
                "city": self.metadata.get("city") or self.metadata.get("location"),
                "station_id": self.metadata.get("station_id"),
            },
            sort_keys=True,
        ).encode("utf-8")
        self.doc_id = hashlib.md5(stable_fingerprint).hexdigest()

class NOAADataManager:
    """Handles NOAA data download and processing - Station-based approach"""

    # Major US cities with their preferred weather stations
    MAJOR_US_CITIES = {
        "New York, NY": ["USW00094728", "USW00014732"],  # Central Park, La Guardia
        "Los Angeles, CA": ["USW00023174", "USW00003167"],  # LAX, Burbank
        "Chicago, IL": ["USW00094846", "USW00014819"],  # O'Hare, Midway
        "Houston, TX": ["USW00012960", "USW00012918"],  # Intercontinental, Hobby
        "Phoenix, AZ": ["USW00023183", "USW00003145"],  # Sky Harbor, Deer Valley
        "Philadelphia, PA": ["USW00013739", "USW00014751"],  # Int'l Airport, Northeast
        "San Antonio, TX": ["USW00012921"],  # International
        "San Diego, CA": ["USW00023188", "USW00003177"],  # Lindbergh, Montgomery
        "Dallas, TX": ["USW00003927", "USW00013960"],  # DFW, Love Field
        "San Jose, CA": ["USW00023293", "USW00023244"],  # Mineta, Reid-Hillview
        "Austin, TX": ["USW00013904", "USW00013958"],  # Bergstrom, Mabry
        "Jacksonville, FL": ["USW00013889", "USW00012815"],  # Int'l, Craig
        "Fort Worth, TX": ["USW00003927"],  # Alliance
        "Columbus, OH": ["USW00014821"],  # Port Columbus
        "San Francisco, CA": ["USW00023234", "USW00023237"],  # Int'l, Oakland
        "Charlotte, NC": ["USW00013881", "USW00053872"],  # Douglas Int'l
        "Indianapolis, IN": ["USW00093819", "USW00014827"],  # Int'l, Regional
        "Seattle, WA": ["USW00024233", "USW00094290"],  # SeaTac, Boeing Field
        "Denver, CO": ["USW00003017", "USW00023062"],  # Int'l, Centennial
        "Washington, DC": ["USW00013743", "USW00093721"],  # Reagan, Dulles
        "Boston, MA": ["USW00014739", "USW00054704"],  # Logan, Norwood
        "El Paso, TX": ["USW00023044", "USW00023045"],  # Int'l, Biggs AAF
        "Detroit, MI": ["USW00094847", "USW00014822"],  # Metro, Coleman Young
        "Nashville, TN": ["USW00013897"],  # Int'l
        "Portland, OR": ["USW00024229", "USW00094261"],  # Int'l, Hillsboro
        "Memphis, TN": ["USW00013893"],  # Int'l
        "Oklahoma City, OK": ["USW00013967", "USW00013964"],  # Will Rogers, Wiley Post
        "Las Vegas, NV": ["USW00023169", "USW00003160"],  # McCarran, Henderson
        "Louisville, KY": ["USW00093821", "USW00013809"],  # Int'l, Bowman
        "Baltimore, MD": ["USW00093721", "USW00093784"],  # BWI, Martin State
        "Milwaukee, WI": ["USW00014839", "USW00004848"],  # Mitchell, Timmerman
        "Albuquerque, NM": ["USW00023050"],  # Int'l Sunport
        "Tucson, AZ": ["USW00023160", "USW00003195"],  # Int'l, Davis-Monthan
        "Fresno, CA": ["USW00093193", "USW00023237"],  # Yosemite, Chandler
        "Sacramento, CA": ["USW00023232", "USW00093225"],  # Int'l, Executive
        "Mesa, AZ": ["USW00003192", "USW00003145"],  # Falcon Field, Phoenix-Mesa
        "Kansas City, MO": ["USW00003947", "USW00013988"],  # Int'l, Downtown
        "Atlanta, GA": ["USW00013874", "USW00093842"],  # Hartsfield, Fulton
        "Miami, FL": ["USW00012839", "USW00012885"],  # Int'l, Opa-locka
        "Cleveland, OH": ["USW00014820", "USW00004853"],  # Hopkins, Burke Lakefront
        "New Orleans, LA": ["USW00012916", "USW00053917"],  # Int'l, Lakefront
        "Minneapolis, MN": ["USW00014922", "USW00014925"],  # St. Paul Int'l, Crystal
        "Tampa, FL": ["USW00012842", "USW00012843"],  # Int'l, Peter O. Knight
        "Orlando, FL": ["USW00012815", "USW00012841"],  # Int'l, Executive
        "Pittsburgh, PA": ["USW00094823", "USW00014762"],  # Int'l, Allegheny
        "Cincinnati, OH": ["USW00093812", "USW00093814"],  # N. Kentucky, Lunken
        "St. Louis, MO": ["USW00013994", "USW00003966"],  # Lambert, Spirit
        "Raleigh, NC": ["USW00013722"],  # Durham Int'l
        "Salt Lake City, UT": ["USW00024127"],  # Int'l
        "Buffalo, NY": ["USW00014733", "USW00004725"]  # Niagara Int'l, Buffalo Airport
    }

    def __init__(self, data_dir: str = "climate_data"):
        self.data_dir = data_dir
        self.station_dir = os.path.join(data_dir, "stations")
        self.station_mapping = {}
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.station_dir, exist_ok=True)

    def download_station_metadata(self) -> Dict[str, Dict]:
        """Download and parse NOAA station metadata"""
        stations_file = os.path.join(self.data_dir, "ghcnd-stations.txt")
        url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"

        # Download if not exists
        if not os.path.exists(stations_file):
            logger.info("Downloading NOAA station metadata...")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(stations_file, 'wb') as f:
                        f.write(response.content)
                    logger.info("✓ Downloaded station metadata")
                else:
                    logger.error(f"Failed to download: HTTP {response.status_code}")
                    return {}
            except Exception as e:
                logger.error(f"Failed to download station metadata: {e}")
                return {}

        # Parse stations
        logger.info("Parsing station metadata...")
        stations = {}

        with open(stations_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(line) < 85:
                    continue

                # Extract fields based on fixed positions
                station_id = line[0:11].strip()
                lat = line[12:20].strip()
                lon = line[21:30].strip()
                elev = line[31:37].strip()
                state = line[38:40].strip()
                name = line[41:71].strip()

                # Clean city name
                city = self._clean_station_name(name)

                stations[station_id] = {
                    "station_id": station_id,
                    "raw_name": name,
                    "city": city,
                    "state": state,
                    "country": station_id[0:2],
                    "latitude": float(lat) if lat else None,
                    "longitude": float(lon) if lon else None,
                    "elevation": float(elev) if elev else None
                }

        self.station_mapping = stations
        logger.info(f"Parsed {len(stations)} stations")

        # Save mapping
        mapping_file = os.path.join(self.data_dir, "station_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(stations, f, indent=2)

        return stations

    def _clean_station_name(self, name: str) -> str:
        """Clean station name to extract city"""
        city = ' '.join(name.split())

        # Remove common suffixes
        patterns = [
            r'\s+AP$', r'\s+INTL\s+AP$', r'\s+INTERNATIONAL$',
            r'\s+AIRPORT$', r'\s+WSO$', r'\s+\d+\s*[NSEW]+$',
            r'\s+AWS$', r'\s+AWOS$'
        ]

        for pattern in patterns:
            city = re.sub(pattern, '', city, flags=re.IGNORECASE)

        return city.title()

    def download_station_data(self, station_id: str) -> bool:
        """Download .dly file for a specific station"""
        filename = f"{station_id}.dly"
        filepath = os.path.join(self.station_dir, filename)

        # Skip if already exists
        if os.path.exists(filepath):
            logger.info(f"✓ {station_id} data already exists")
            return True

        url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{filename}"

        logger.info(f"Downloading {station_id} data...")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✓ Downloaded {station_id} ({len(response.content)/1024/1024:.1f}MB)")
                return True
            else:
                logger.error(f"Failed to download {station_id}: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to download {station_id}: {e}")
            return False

    def download_major_cities_data(self, cities: Optional[List[str]] = None) -> bool:
        """Download station data for major US cities"""
        if cities is None:
            cities = list(self.MAJOR_US_CITIES.keys())

        logger.info(f"Downloading data for {len(cities)} cities")

        success_count = 0
        total_stations = 0

        for city in cities:
            if city not in self.MAJOR_US_CITIES:
                logger.warning(f"City '{city}' not in major cities list")
                continue

            station_ids = self.MAJOR_US_CITIES[city]
            logger.info(f"\nProcessing {city} ({len(station_ids)} stations)")

            for station_id in station_ids:
                total_stations += 1
                if self.download_station_data(station_id):
                    success_count += 1

        logger.info(f"\n✓ Downloaded {success_count}/{total_stations} stations successfully")
        return success_count > 0

    def parse_dly_file(self, filepath: str, start_year: Optional[int] = None,
                      end_year: Optional[int] = None) -> List[ClimateDocument]:
        """Parse a .dly station file into ClimateDocuments"""
        documents = []
        station_id = os.path.basename(filepath).replace('.dly', '')

        if station_id not in self.station_mapping:
            logger.warning(f"Station {station_id} not found in metadata")
            return documents

        station_info = self.station_mapping[station_id]
        city_name = None

        # Find city name from our mapping
        for city, stations in self.MAJOR_US_CITIES.items():
            if station_id in stations:
                city_name = city
                break

        if not city_name:
            city_name = f"{station_info['city']}, {station_info['state']}"

        logger.info(f"Parsing {station_id} for {city_name}")

        # Track daily data to aggregate
        daily_data = defaultdict(lambda: defaultdict(dict))

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(line) < 269:  # Minimum valid line length
                    continue

                # Parse fixed-width format
                year = int(line[11:15])
                month = int(line[15:17])
                element = line[17:21].strip()

                # Filter by year range if specified
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue

                # Only process key elements
                if element not in ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'SNOW']:
                    continue

                # Parse daily values
                for day in range(1, 32):
                    # Each day: 5 chars for value, 3 for flags
                    start_pos = 21 + (day - 1) * 8
                    if start_pos + 5 > len(line):
                        break

                    value_str = line[start_pos:start_pos + 5].strip()

                    # Skip missing values
                    if not value_str or value_str == '-9999':
                        continue

                    try:
                        value = int(value_str)
                        date_obj = datetime(year, month, day).date()
                        date_key = str(date_obj)

                        # Store in daily_data for aggregation
                        daily_data[date_key][element] = value

                    except (ValueError, OverflowError):
                        continue

        # Convert aggregated daily data to documents
        for date_str, measurements in daily_data.items():
            # Skip if no temperature data
            if not any(k in measurements for k in ['TMAX', 'TMIN', 'TAVG']):
                continue

            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()

            # Build content
            content_parts = [
                f"Date: {date_obj}",
                f"Location: {city_name}",
                f"Station: {station_info['raw_name']}"
            ]

            metadata = {
                'station_id': station_id,
                'city': city_name,
                'state': station_info['state'],
                'country': 'US',
                'date': date_str,
                'year': date_obj.year,
                'month': date_obj.month,
                'latitude': station_info['latitude'],
                'longitude': station_info['longitude']
            }

            # Add measurements with proper scaling
            if 'TMAX' in measurements:
                tmax = measurements['TMAX'] / 10  # Convert to Celsius
                content_parts.append(f"Max Temperature: {tmax:.1f}°C ({tmax * 1.8 + 32:.1f}°F)")
                metadata['tmax'] = tmax

            if 'TMIN' in measurements:
                tmin = measurements['TMIN'] / 10
                content_parts.append(f"Min Temperature: {tmin:.1f}°C ({tmin * 1.8 + 32:.1f}°F)")
                metadata['tmin'] = tmin

            if 'TAVG' in measurements:
                tavg = measurements['TAVG'] / 10
                content_parts.append(f"Avg Temperature: {tavg:.1f}°C ({tavg * 1.8 + 32:.1f}°F)")
                metadata['tavg'] = tavg
            elif 'TMAX' in measurements and 'TMIN' in measurements:
                # Calculate average if not provided
                tavg = (measurements['TMAX'] + measurements['TMIN']) / 20
                content_parts.append(f"Avg Temperature: {tavg:.1f}°C ({tavg * 1.8 + 32:.1f}°F)")
                metadata['tavg'] = tavg

            if 'PRCP' in measurements:
                prcp = measurements['PRCP'] / 10  # Convert to mm
                content_parts.append(f"Precipitation: {prcp:.1f}mm ({prcp / 25.4:.2f}in)")
                metadata['prcp'] = prcp

            if 'SNOW' in measurements:
                snow = measurements['SNOW']  # Already in mm
                content_parts.append(f"Snowfall: {snow}mm ({snow / 25.4:.1f}in)")
                metadata['snow'] = snow

            # Create document
            doc = ClimateDocument(
                content='\n'.join(content_parts),
                source='NOAA',
                doc_type='historical_weather',
                metadata=metadata
            )
            documents.append(doc)

        logger.info(f"✓ Parsed {len(documents)} daily records from {station_id}")
        return documents

    def process_station_data(self, cities: Optional[List[str]] = None,
                           start_year: Optional[int] = None,
                           end_year: Optional[int] = None) -> List[ClimateDocument]:
        """Process downloaded station data into ClimateDocuments"""
        if not self.station_mapping:
            self.download_station_metadata()

        if cities is None:
            cities = list(self.MAJOR_US_CITIES.keys())

        all_documents = []

        for city in cities:
            if city not in self.MAJOR_US_CITIES:
                continue

            logger.info(f"\nProcessing data for {city}")
            station_ids = self.MAJOR_US_CITIES[city]

            for station_id in station_ids:
                filepath = os.path.join(self.station_dir, f"{station_id}.dly")
                if not os.path.exists(filepath):
                    logger.warning(f"Data file for {station_id} not found")
                    continue

                documents = self.parse_dly_file(filepath, start_year, end_year)
                all_documents.extend(documents)

        logger.info(f"\n✓ Total documents created: {len(all_documents)}")
        return all_documents

class ClimateDataCollector:
    """Collects current weather data from APIs"""

    def __init__(self):
        self.geocoder = Nominatim(user_agent="climate_rag_v2")

    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location"""
        try:
            result = self.geocoder.geocode(location, timeout=10)
            if result:
                return (result.latitude, result.longitude)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding error for {location}: {e}")
        return None

    def collect_meteostat_data(self, location: str, days_back: int = 30) -> List[ClimateDocument]:
        """Collect recent historical data from Meteostat"""
        coords = self.get_coordinates(location)
        if not coords:
            logger.warning(f"Could not geocode {location}")
            return []

        documents = []

        try:

            point = Point(coords[0], coords[1])

            end_date = datetime.now()

            start_date = end_date - timedelta(days=days_back)

            data = Daily(point, start_date, end_date)

            data = data.fetch()

            for idx, row in data.iterrows():
                content_parts = [
                    f"Date: {idx.date()}",
                    f"Location: {location}",
                    f"Coordinates: {coords[0]:.4f}, {coords[1]:.4f}"
                ]

                metadata = {
                    'location': location,
                    'latitude': coords[0],
                    'longitude': coords[1],
                    'date': str(idx.date()),
                    'year': idx.year,
                    'month': idx.month
                }

                # Add available measurements
                if pd.notna(row.get('tavg')):
                    content_parts.append(f"Avg Temperature: {row['tavg']:.1f}°C")
                    metadata['tavg'] = row['tavg']

                if pd.notna(row.get('tmin')):
                    content_parts.append(f"Min Temperature: {row['tmin']:.1f}°C")
                    metadata['tmin'] = row['tmin']

                if pd.notna(row.get('tmax')):
                    content_parts.append(f"Max Temperature: {row['tmax']:.1f}°C")
                    metadata['tmax'] = row['tmax']

                if pd.notna(row.get('prcp')):
                    content_parts.append(f"Precipitation: {row['prcp']:.1f}mm")
                    metadata['prcp'] = row['prcp']

                doc = ClimateDocument(
                    content='\n'.join(content_parts),
                    source='Meteostat',
                    doc_type='recent_weather',
                    metadata=metadata
                )
                documents.append(doc)

            logger.info(f"✓ Collected {len(documents)} days from Meteostat for {location}")

        except Exception as e:
            logger.error(f"Error collecting Meteostat data: {e}")

        return documents

    def collect_openmeteo_data(self, location: str) -> List[ClimateDocument]:
        """Collect current and forecast data from Open-Meteo"""
        coords = self.get_coordinates(location)
        if not coords:
            logger.warning(f"Could not geocode {location}")
            return []

        documents = []

        try:
            url = "https://api.open-meteo.com/v1/forecast"

            params = {
                "latitude": coords[0],
                "longitude": coords[1],
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 7
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()

                # Current conditions
                current = data.get('current', {})
                content_parts = [
                    f"Time: {current.get('time', 'N/A')}",
                    f"Location: {location}",
                    f"Coordinates: {coords[0]:.4f}, {coords[1]:.4f}",
                    f"Current Temperature: {current.get('temperature_2m', 'N/A')}°C",
                    f"Humidity: {current.get('relative_humidity_2m', 'N/A')}%",
                    f"Precipitation: {current.get('precipitation', 0)}mm",
                    f"Wind Speed: {current.get('wind_speed_10m', 'N/A')}km/h"
                ]

                doc = ClimateDocument(
                    content='\n'.join(content_parts),
                    source='OpenMeteo',
                    doc_type='current_weather',
                    metadata={
                        'location': location,
                        'latitude': coords[0],
                        'longitude': coords[1],
                        'temperature': current.get('temperature_2m'),
                        'humidity': current.get('relative_humidity_2m'),
                        'precipitation': current.get('precipitation'),
                        'wind_speed': current.get('wind_speed_10m')
                    }
                )
                documents.append(doc)

                # Daily forecast
                daily = data.get('daily', {})
                if daily:
                    for i in range(len(daily.get('time', []))):
                        content_parts = [
                            f"Date: {daily['time'][i]}",
                            f"Location: {location}",
                            f"Max Temperature: {daily['temperature_2m_max'][i]}°C",
                            f"Min Temperature: {daily['temperature_2m_min'][i]}°C",
                            f"Precipitation: {daily['precipitation_sum'][i]}mm"
                        ]

                        doc = ClimateDocument(
                            content='\n'.join(content_parts),
                            source='OpenMeteo',
                            doc_type='weather_forecast',
                            metadata={
                                'location': location,
                                'latitude': coords[0],
                                'longitude': coords[1],
                                'date': daily['time'][i],
                                'tmax': daily['temperature_2m_max'][i],
                                'tmin': daily['temperature_2m_min'][i],
                                'precipitation': daily['precipitation_sum'][i]
                            }
                        )
                        documents.append(doc)

                logger.info(f"✓ Collected current + {len(documents)-1} forecast days from Open-Meteo for {location}")

        except Exception as e:
            logger.error(f"Error collecting Open-Meteo data: {e}")

        return documents

class VectorStore:
    """FAISS-based vector store with hybrid retrieval"""

    def __init__(self, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = 384  # MiniLM dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self.bm25 = None

    def _build_enhanced_query(self, query: str, year: Optional[int] = None, month: Optional[int] = None) -> str:
        """Build enhanced query with temporal context"""

        # Start with original query
        query_parts = [query]

        # Month names for better semantic matching
        month_names = {
            1: "january", 2: "february", 3: "march", 4: "april",
            5: "may", 6: "june", 7: "july", 8: "august",
            9: "september", 10: "october", 11: "november", 12: "december"
        }

        # Season mapping for additional context
        seasons = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall"
        }

        # Add temporal information
        if year is not None and month is not None:
            # Full date context: "July 2023 summer"
            month_name = month_names.get(month, str(month))
            season = seasons.get(month, "")
            query_parts.extend([month_name, str(year), season])

        elif year is not None:
            # Year only: "2023"
            query_parts.append(str(year))

        elif month is not None:
            # Month only: "July summer"
            month_name = month_names.get(month, str(month))
            season = seasons.get(month, "")
            query_parts.extend([month_name, season])

        # Join with spaces and clean up
        enhanced_query = " ".join(filter(None, query_parts))

        return enhanced_query

    def build_index(self, documents: List[ClimateDocument]):
        """Build FAISS index and BM25 for hybrid retrieval"""
        logger.info(f"Building index for {len(documents)} documents...")

        # Extract texts
        texts = [doc.content for doc in documents]
        self.documents = documents

        # Create embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=128)
        embeddings = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        if len(documents) < 10000:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # Use IVF for larger datasets
            nlist = int(np.sqrt(len(documents)))
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)

        self.index.add(embeddings)

        # Build BM25 for sparse retrieval
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)

        logger.info(f"✓ Index built with {self.index.ntotal} vectors")

    def hybrid_search(self, query: str, locations, top_k: int = 200, alpha: float = 0.7, year: Optional[int] = None, month: Optional[int] = None) -> List[Tuple[ClimateDocument, float]]:
        """Hybrid search combining dense and sparse retrieval with optional date filtering"""
        if not self.index or not self.documents:
            logger.warning("No index built yet")
            return []

        if len(locations) > 1:
            # Multi-location query - search for each location separately
            return self._multi_location_search(query, locations, top_k, alpha, year, month)
        else:
            # Single location or no specific locations - use original method
            return self._single_location_search(query, top_k, alpha, year, month)

    def _multi_location_search(self, query: str, locations: List[str], top_k: int, alpha: float, year: Optional[int] = None, month: Optional[int] = None) -> List[Tuple[ClimateDocument, float]]:
        """Handle multi-location queries by searching each location separately"""

        all_results = []
        per_location_k = max(10, top_k // len(locations))  # Distribute top_k across locations

        for location in locations:

            # Create location-specific query
            location_query = f"{query} {location}".strip()

            # Search with location filter
            location_results = self._search_with_location_filter(location_query, location, per_location_k, alpha, year, month)

            all_results.extend(location_results)

        # Sort all results by score and return top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _search_with_location_filter(self, query: str, target_location: str, top_k: int, alpha: float, year: Optional[int] = None, month: Optional[int] = None) -> List[Tuple[ClimateDocument, float]]:
        """Search with location filtering"""

        # Enhanced query for better matching
        enhanced_query = query

        # Dense search
        query_embedding = self.embedder.encode([enhanced_query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = self.index.search(query_embedding, len(self.documents))

        # Sparse search
        query_tokens = word_tokenize(enhanced_query.lower())
        sparse_scores = self.bm25.get_scores(query_tokens)
        max_sparse_score = max(sparse_scores) if max(sparse_scores) > 0 else 1.0

        # Combine scores
        combined_scores = {}
        for idx, score in zip(dense_indices[0], dense_scores[0]):
            if idx != -1:
                combined_scores[idx] = alpha * score
        for idx, sparse_score in enumerate(sparse_scores):
            norm_score = sparse_score / max_sparse_score
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * norm_score
            else:
                combined_scores[idx] = (1 - alpha) * norm_score

        # Apply location filter
        location_filtered_scores = {}
        for idx, score in combined_scores.items():
            doc = self.documents[idx]
            doc_location = doc.metadata.get('city', '')

            # Check if document matches target location
            if self._location_matches(doc_location, target_location):
                # Apply date filters
                if year and doc.metadata.get('year') != year:
                    continue
                if month and doc.metadata.get('month') != month:
                    continue
                location_filtered_scores[idx] = score

        # Return top results for this location
        sorted_results = sorted(location_filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[idx], score) for idx, score in sorted_results]

    def _location_matches(self, doc_location: str, target_location: str) -> bool:
        """Check if document location matches target location"""
        if not doc_location or not target_location:
            return False

        # Normalize for comparison
        doc_loc = doc_location.lower().strip()
        target_loc = target_location.lower().strip()

        # Direct match
        if doc_loc == target_loc:
            return True

        # Check if target is contained in doc location
        if target_loc in doc_loc:
            return True

        # Check city name without state (Phoenix vs Phoenix, AZ)
        doc_city = doc_loc.split(',')[0].strip()
        target_city = target_loc.split(',')[0].strip()

        return doc_city == target_city

    def _single_location_search(self, query: str, top_k: int, alpha: float, year: Optional[int] = None, month: Optional[int] = None) -> List[Tuple[ClimateDocument, float]]:
        """Single location search method with enhanced query and consistent filtering"""

        enhanced_query = query

        # Dense search
        query_embedding = self.embedder.encode([enhanced_query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = self.index.search(query_embedding, len(self.documents))

        # Sparse search
        query_tokens = word_tokenize(enhanced_query.lower())
        sparse_scores = self.bm25.get_scores(query_tokens)
        max_sparse_score = max(sparse_scores) if max(sparse_scores) > 0 else 1.0

        # Combine scores for all docs
        combined_scores = {}
        for idx, score in zip(dense_indices[0], dense_scores[0]):
            if idx != -1:
                combined_scores[idx] = alpha * score
        for idx, sparse_score in enumerate(sparse_scores):
            norm_score = sparse_score / max_sparse_score
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * norm_score
            else:
                combined_scores[idx] = (1 - alpha) * norm_score

        # Apply filtering like multi-location search
        filtered_scores = {}
        for idx, score in combined_scores.items():
            doc = self.documents[idx]

            # Apply date filters (same as multi-location search)
            if year and doc.metadata.get('year') != year:
                continue
            if month and doc.metadata.get('month') != month:
                continue

            filtered_scores[idx] = score

        # Sort all results by score and apply top_k
        sorted_results = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[idx], score) for idx, score in sorted_results]

    def save(self, path: str):
        """Save index and data"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        faiss.write_index(self.index, f"{path}_index.faiss")
        with open(f"{path}_data.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'bm25': self.bm25
            }, f)
        logger.info(f"✓ Saved index to {path}")

    def load(self, path: str):
        """Load index and data"""
        self.index = faiss.read_index(f"{path}_index.faiss")
        with open(f"{path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.bm25 = data['bm25']
        logger.info(f"Loaded index with {len(self.documents)} documents")

#from openai import OpenAI
import google.generativeai as genai

class ClimateRAG:
    """Main RAG system for climate analysis"""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.vector_store = VectorStore()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Initialize Gemini
        self.gemini_api_key = gemini_api_key
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

        # Data managers
        self.noaa_manager = NOAADataManager()
        self.data_collector = ClimateDataCollector()

        # Location extractor
        self.location_cache = {}

    def initialize_with_cities(self, cities: Optional[List[str]] = None,
                               start_year: Optional[int] = None,
                               end_year: Optional[int] = None):
        """Initialize with station data for specific cities"""
        logger.info("Initializing with city-based station data...")

        # Download station metadata
        self.noaa_manager.download_station_metadata()

        # Download station data for cities
        self.noaa_manager.download_major_cities_data(cities)

        # Process into documents
        documents = self.noaa_manager.process_station_data(cities, start_year, end_year)

        if documents:
            self.vector_store.build_index(documents)
            logger.info(f"Initialized with {len(documents)} historical documents")
        else:
            logger.warning("No historical documents created")

    def add_current_data(self, locations: List[str]):
        """Add current weather data for locations"""
        logger.info(f"Adding current data for {len(locations)} locations...")

        all_docs = []

        for location in locations:
            # Meteostat recent data
            docs = self.data_collector.collect_meteostat_data(location)
            all_docs.extend(docs)

            # Open-Meteo current + forecast
            docs = self.data_collector.collect_openmeteo_data(location)
            all_docs.extend(docs)

        if all_docs:
            # Rebuild index with all documents
            existing_docs = self.vector_store.documents if self.vector_store.documents else []
            all_documents = existing_docs + all_docs
            self.vector_store.build_index(all_documents)
            logger.info(f"✓ Added {len(all_docs)} current weather documents")

    def extract_locations(self, text: str) -> List[str]:
        """Extract location names from text using NER"""
        doc = nlp(text)
        locations = []

        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                location = ent.text.strip()

                # Validate with geocoding (with caching)
                if location not in self.location_cache:
                    coords = self.data_collector.get_coordinates(location)
                    self.location_cache[location] = coords is not None

                if self.location_cache[location]:
                    locations.append(location)

        return list(set(locations))  # Remove duplicates

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        logger.info(f"Query: {question}")

        # Extract locations from query
        locations = self.extract_locations(question)
        if locations:
            logger.info(f"Detected locations: {locations}")

        # --- Extract date from query FIRST ---
        import re
        date_match = re.search(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",
            question, re.IGNORECASE
        )

        target_month = None
        target_year = None

        if date_match:
            month_str, year_str = date_match.groups()

            try:
                # Fix the month parsing - use full month name
                months = ['january', 'february', 'march', 'april', 'may', 'june',
                        'july', 'august', 'september', 'october', 'november', 'december']
                target_month = months.index(month_str.lower()) + 1
                target_year = int(year_str)
                logger.info(f"Looking for {month_str.title()} {year_str} (month={target_month})")
            except ValueError as e:
                logger.warning(f"Failed to parse date: {e}")

        # Hybrid search with date parameters - returns ALL matching documents
        search_results = self.vector_store.hybrid_search(
            question,
            locations,
            year=target_year,
            month=target_month
        )

        documents = [doc for doc, _ in search_results]

        if not search_results:
            return {
                'answer': "I don't have enough information to answer that question.",
                'sources': [],
                'locations': locations
            }

        final_docs = [doc for doc, _ in search_results]

        # Build context
        context_parts = []
        for doc in final_docs:
            context_parts.append(f"[Source: {doc.source}]\n{doc.content}")
        context = "\n\n".join(context_parts)

        # Generate answer
        #if self.openai_api_key:
        if self.gemini_api_key:
            answer = self._generate_answer(question, context)
        else:
            answer = self._simple_answer(question, final_docs)

        return {
            'answer': answer,
            'sources': [{'content': doc.content, 'metadata': doc.metadata} for doc in final_docs],
            'locations': locations,
            'context': context
        }

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Gemini Flash"""
        try:
            prompt = f"""You are a helpful climate data assistant. Answer questions based on the provided context.
            Be specific and cite data when available. If the context is empty, answer based on your general knowledge.
            Use natural language and avoid tables or complex formatting. Make your response easy to read and conversational.
            Output only relevant data even if you have more information. Give only what the user is asking for from the data that you receive. Nothing more. Nothing less.
            If you don't find the exact data, give whatever you have.

            Context:
            {context}

            Question: {question}

            Please provide a comprehensive answer based on the available data."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=512,
                    top_p=0.9,
                )
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return None

    def _simple_answer(self, question: str, documents: List[ClimateDocument]) -> str:
        """Simple answer without LLM"""
        if not documents:
            return "No relevant information found."

        # Extract key information
        answer_parts = [f"Based on the available data:"]

        for doc in documents[:3]:  # Use top 3 documents
            answer_parts.append(f"\n- {doc.content}")

        return "\n".join(answer_parts)

    def save(self, path: str = "climate_rag"):
        """Save the system"""
        self.vector_store.save(path)

    def load(self, path: str = "climate_rag"):
        """Load the system"""
        self.vector_store.load(path)

def load_or_initialize_rag(TOP_50_CITIES, GEMINI_API_KEY, save_path="climate_rag_50_cities"):
    """Load existing RAG or initialize if not found"""
    rag = ClimateRAG(GEMINI_API_KEY)

    # Check if saved files exist
    if (os.path.exists(f"{save_path}_index.faiss") and
        os.path.exists(f"{save_path}_data.pkl")):

        print("✓ Found existing index, loading...")
        rag.load(save_path)
        print(f"Loaded {len(rag.vector_store.documents)} documents")

    else:
        print("No existing index found, initializing...")
        # One-time initialization
        rag.initialize_with_cities(TOP_50_CITIES, start_year=2020, end_year=2024)
        rag.add_current_data(TOP_50_CITIES)
        rag.save(save_path)

    return rag

# Imports for Evaluation
import re
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    query: str
    expected_answer: Any
    retrieved_answer: Any
    sources_used: List[Dict]
    response_time: float
    is_correct: bool
    error_type: Optional[str] = None
    metadata: Dict = None


class ClimateRAGEvaluator:
    """Comprehensive evaluation suite for Climate RAG system"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.evaluation_results = []

    def evaluate_factual_accuracy(self, test_queries: List[Dict]) -> Dict[str, float]:
        """
        Evaluate exact match and numerical accuracy

        test_queries format:
        [
            {
                'query': 'What was the temperature in Phoenix on July 15, 2023?',
                'expected_value': 42.3,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Phoenix, AZ',
                'date': '2023-07-15'
            }
        ]
        """
        numerical_matches = 0
        total_queries = len(test_queries)

        results = []

        for test_case in test_queries:
            start_time = time.time()

            # Query the RAG system
            response = self.rag_system.query(test_case['query'])

            response_time = time.time() - start_time

            print("response:", response)
            print("response['answer']:", response['answer'])

            # Extract numerical values from response
            retrieved_value = self._extract_numerical_value(
                response['answer'],
                test_case['query_type']
            )
            print("retrieved_value: ", retrieved_value)
            expected_value = test_case['expected_value']
            print("expected_value: ", expected_value)

            # Numerical accuracy within tolerance
            is_numerical_match = self._is_within_tolerance(
                retrieved_value,
                expected_value,
                test_case['query_type']
            )
            if is_numerical_match:
                numerical_matches += 1

            # Store detailed results
            eval_result = EvaluationResult(
                query=test_case['query'],
                expected_answer=expected_value,
                retrieved_answer=retrieved_value,
                sources_used=response.get('sources', []),
                response_time=response_time,
                is_correct=is_numerical_match,
                metadata=test_case
            )

            results.append(eval_result)

        self.evaluation_results.extend(results)

        return {
            'numerical_accuracy': numerical_matches / total_queries,
            'total_queries': total_queries,
            'numerical_matches': numerical_matches
        }

    def evaluate_temporal_precision(self, temporal_test_cases: List[Dict]) -> Dict[str, float]:
        """
        Evaluate temporal filtering accuracy

        temporal_test_cases format:
        [
            {
                'query': 'Phoenix temperature July 2023',
                'expected_month': 7,
                'expected_year': 2023,
                'should_exclude_dates': ['2023-06-30', '2023-08-01']
            }
        ]
        """
        correct_temporal_filtering = 0
        total_cases = len(temporal_test_cases)

        results = []

        for test_case in temporal_test_cases:
            start_time = time.time()

            response = self.rag_system.query(test_case['query'])
            response_time = time.time() - start_time

            # Check if retrieved sources match expected temporal constraints
            sources = response.get('sources', [])

            temporal_accuracy = self._check_temporal_constraints(
                sources,
                test_case.get('expected_year'),
                test_case.get('expected_month'),
                test_case.get('should_exclude_dates', [])
            )

            if temporal_accuracy:
                correct_temporal_filtering += 1

            eval_result = EvaluationResult(
                query=test_case['query'],
                expected_answer=f"Year: {test_case.get('expected_year')}, Month: {test_case.get('expected_month')}",
                retrieved_answer=f"Sources from correct time period: {temporal_accuracy}",
                sources_used=sources,
                response_time=response_time,
                is_correct=temporal_accuracy,
                metadata=test_case
            )

            results.append(eval_result)

        self.evaluation_results.extend(results)

        return {
            'temporal_precision': correct_temporal_filtering / total_cases,
            'total_cases': total_cases,
            'correct_filtering': correct_temporal_filtering
        }

    def evaluate_response_time(self, performance_queries: List[str]) -> Dict[str, float]:
        """Evaluate system response time performance"""
        response_times = []

        for query in performance_queries:
            start_time = time.time()
            response = self.rag_system.query(query)
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

        return {
            'average_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'total_queries': len(performance_queries)
        }

    def evaluate_coverage_metrics(self) -> Dict[str, Any]:
        """Evaluate data coverage across cities and dates"""
        documents = self.rag_system.vector_store.documents

        # City coverage
        cities = set()
        years = set()
        months = set()
        sources = set()

        for doc in documents:
            metadata = doc.metadata
            if 'city' in metadata:
                cities.add(metadata['city'])
            if 'year' in metadata:
                years.add(metadata['year'])
            if 'month' in metadata:
                months.add(metadata['month'])
            sources.add(doc.source)

        # Calculate coverage gaps
        expected_years = set(range(2020, 2025))  # Based on your system
        year_coverage = len(years & expected_years) / len(expected_years)

        expected_months = set(range(1, 13))
        month_coverage = len(months & expected_months) / len(expected_months)

        return {
            'total_documents': len(documents),
            'unique_cities': len(cities),
            'cities_covered': list(cities),
            'year_range': f"{min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}",
            'year_coverage_ratio': year_coverage,
            'month_coverage_ratio': month_coverage,
            'data_sources': list(sources),
            'documents_by_source': {source: sum(1 for doc in documents if doc.source == source) for source in sources}
        }

    def run_comprehensive_evaluation(self, test_suite: Dict[str, List]) -> Dict[str, Any]:
        """Run all evaluation metrics"""
        print("Running comprehensive Climate RAG evaluation...")

        results = {}

        # 1. Factual Accuracy
        if 'factual_queries' in test_suite:
            print("Evaluating factual accuracy...")
            results['factual_accuracy'] = self.evaluate_factual_accuracy(test_suite['factual_queries'])

        # 2. Temporal Precision
        if 'temporal_queries' in test_suite:
            print("Evaluating temporal precision...")
            results['temporal_precision'] = self.evaluate_temporal_precision(test_suite['temporal_queries'])

        # 3. Response Time
        if 'performance_queries' in test_suite:
            print("⚡ Evaluating response time...")
            results['response_time'] = self.evaluate_response_time(test_suite['performance_queries'])

        # 4. Coverage Metrics
        print("Evaluating coverage metrics...")
        results['coverage_metrics'] = self.evaluate_coverage_metrics()

        # Overall summary
        results['overall_summary'] = self._generate_overall_summary(results)

        return results

    # Helper methods
    def _extract_numerical_value(self, text: str, query_type: str) -> Optional[float]:
        """Extract numerical values from response text"""
        if query_type == 'temperature':
            # Look for temperature patterns: "42.3°C", "108.1°F", "42.3 degrees"
            temp_patterns = [
                r'(-?\d+\.?\d*)\s*°C',
                r'(-?\d+\.?\d*)\s*°F',
                r'(-?\d+\.?\d*)\s*degrees?\s*(?:celsius|fahrenheit|C|F)?'
            ]
        elif query_type == 'precipitation':
            # Look for precipitation patterns: "15.2mm", "0.6 inches"
            temp_patterns = [
                r'(-?\d+\.?\d*)\s*mm',
                r'(-?\d+\.?\d*)\s*inches?',
                r'(-?\d+\.?\d*)\s*in'
            ]
        else:
            # Generic number extraction
            temp_patterns = [r'(-?\d+\.?\d*)']

        for pattern in temp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _is_within_tolerance(self, retrieved_value: Optional[float], expected_value: float, query_type: str) -> bool:
        """Check if retrieved value is within acceptable tolerance"""
        if retrieved_value is None:
            return False

        tolerance = {
            'temperature': 0.5,  # ±0.5°C
            'precipitation': 1.0,  # ±1mm
            'default': 0.1  # ±0.1 for generic numbers
        }

        tol = tolerance.get(query_type, tolerance['default'])
        return abs(retrieved_value - expected_value) <= tol

    def _check_temporal_constraints(self, sources: List[Dict], expected_year: Optional[int],
                                  expected_month: Optional[int], excluded_dates: List[str]) -> bool:
        """Check if sources match temporal constraints"""
        if not sources:
            return False

        for source in sources:
            metadata = source.get('metadata', {})

            # Check year constraint
            if expected_year and metadata.get('year') != expected_year:
                return False

            # Check month constraint
            if expected_month and metadata.get('month') != expected_month:
                return False

            # Check excluded dates
            source_date = metadata.get('date')
            if source_date in excluded_dates:
                return False

        return True

    def _check_location_identification(self, detected_locations: List[str], expected_locations: List[str]) -> bool:
        """Check if all expected locations were detected"""
        if len(detected_locations) != len(expected_locations):
            return False

        # Normalize location names for comparison
        detected_normalized = [self._normalize_location(loc) for loc in detected_locations]
        expected_normalized = [self._normalize_location(loc) for loc in expected_locations]

        return set(detected_normalized) == set(expected_normalized)

    def _normalize_location(self, location: str) -> str:
        """Normalize location names for comparison"""
        return location.lower().strip().replace(',', '').replace('.', '')

    def _check_time_period_consistency(self, sources: List[Dict], expected_period: Dict) -> bool:
        """Check if all sources are from the same expected time period"""
        if not sources:
            return False

        for source in sources:
            metadata = source.get('metadata', {})

            if expected_period.get('year') and metadata.get('year') != expected_period['year']:
                return False

            if expected_period.get('month') and metadata.get('month') != expected_period['month']:
                return False

        return True

    def _classify_comparison_error(self, locations_correct: bool, time_consistent: bool) -> Optional[str]:
        """Classify the type of error in comparison queries"""
        if locations_correct and time_consistent:
            return None
        elif not locations_correct and time_consistent:
            return "location_error"
        elif locations_correct and not time_consistent:
            return "temporal_error"
        else:
            return "both_location_and_temporal_error"

    def _generate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall evaluation summary"""
        summary = {
            'total_evaluation_queries': len(self.evaluation_results),
            'overall_accuracy': sum(1 for r in self.evaluation_results if r.is_correct) / len(self.evaluation_results) if self.evaluation_results else 0
        }

        # Add key metrics if available
        if 'factual_accuracy' in results:
            summary['factual_accuracy_score'] = results['factual_accuracy']['numerical_accuracy']

        if 'temporal_precision' in results:
            summary['temporal_precision_score'] = results['temporal_precision']['temporal_precision']

        if 'multi_location_accuracy' in results:
            summary['multi_location_accuracy_score'] = results['multi_location_accuracy']['multi_location_accuracy']

        if 'response_time' in results:
            summary['avg_response_time'] = results['response_time']['average_response_time']
            summary['p95_response_time'] = results['response_time']['p95_response_time']

        return summary


# Example usage and test data creation
def create_sample_test_suite():
    """Create sample test suite for evaluation"""
    return {
        'factual_queries': [
            {
                'query': 'Seattle rainfall on March 2022',
                'expected_value': 3.8,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Seattle, WA',
                'date': '2022-03-10'
            },
            # Temperature Queries - Specific Dates
            {
                'query': 'How hot was it in Phoenix in July 2023? What is the maximum temperature on July 15?',
                'expected_value': 48.3,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Phoenix, AZ',
                'date': '2023-07-15'
            },
            {
                'query': 'How cold was it in Chicago in January 2024? What is the minimum temperature on January 10?',
                'expected_value': -0.5,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Chicago, IL',
                'date': '2024-01-10'
            },
            {
                'query': 'What was the temperature in Miami in December 2022? What is the value on December 25?',
                'expected_value': 28.3,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Miami, FL',
                'date': '2022-12-25'
            },

            # Precipitation Queries - Specific Dates
            {
                'query': 'How much did it rain in Seattle in March 2022? What is the rainfall on March 10?',
                'expected_value': 3.0,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Seattle, WA',
                'date': '2022-03-10'
            },
            {
                'query': 'How much did it rain in Portland in November 2023? What is the rainfall on November 15?',
                'expected_value': 23.9,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Portland, OR',
                'date': '2023-11-15'
            },
            {
                'query': 'How much did it rain in Boston in September 2021? What is the precipitation on September 8?',
                'expected_value': 0.0,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Boston, MA',
                'date': '2021-09-08'
            },

            # Monthly Average Queries
            {
                'query': 'What was the average temperature in Denver in January 2023?',
                'expected_value': -20.1,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Denver, CO',
                'date': '2023-01'
            },
            {
                'query': 'How much did it rain in Atlanta in March 2023? What was the average rainfall?',
                'expected_value': 138.1,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Atlanta, GA',
                'date': '2023-03'
            },
            {
                'query': 'What was the total precipitation in Minneapolis in December 2022?',
                'expected_value': 44.5,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Minneapolis, MN',
                'date': '2022-12'
            },

            # Extreme Weather Queries
            {
                'query': 'How hot was it in Las Vegas in August 2023? What was the highest temperature recorded?',
                'expected_value': 42.8,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Las Vegas, NV',
                'date': '2023-08'
            },
            {
                'query': 'How cold was it in Buffalo in February 2024? What was the coldest temperature?',
                'expected_value': -12.1,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Buffalo, NY',
                'date': '2024-02'
            },
            {
                'query': 'How much did it rain in New Orleans in June 2023? What was the maximum daily rainfall?',
                'expected_value': 38.5,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'New Orleans, LA',
                'date': '2023-06'
            },

            # Winter/Snow Queries
            {
                'query': 'How cold was it in Detroit in February 2024? What is the minimum temperature on February 14?',
                'expected_value': -1.0,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Detroit, MI',
                'date': '2024-02-14'
            },
            {
                'query': 'How much did it snow in Cleveland in January 2023? What is the snowfall on January 20?',
                'expected_value': 2023.0,
                'expected_unit': 'millimeters',
                'query_type': 'snow',
                'location': 'Cleveland, OH',
                'date': '2023-01-20'
            },
            {
                'query': 'How much did it snow in Colorado Springs in January 2024? What is the snowfall on January 8?',
                'expected_value': 8.0,
                'expected_unit': 'millimeters',
                'query_type': 'snow',
                'location': 'Colorado Springs, CO',
                'date': '2024-01-08'
            },

            # Summer Heat Queries
            {
                'query': 'How hot was it in Phoenix in June 2023? What is the temperature on June 20?',
                'expected_value': 44.4,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Phoenix, AZ',
                'date': '2023-06-20'
            },
            {
                'query': 'How hot was it in Tucson in August 2023? What is the high temperature on August 15?',
                'expected_value': 38.3,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Tucson, AZ',
                'date': '2023-08-15'
            },
            {
                'query': 'How hot was it in Phoenix in July 2023? What is the value on July 1?',
                'expected_value': 45.6,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Phoenix, AZ',
                'date': '2023-07-01'
            },

            # Coastal/Humid Climate Queries
            {
                'query': 'What was the temperature in San Francisco in October 2023? What is the value on October 31?',
                'expected_value': 18.3,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'San Francisco, CA',
                'date': '2023-10-31'
            },
            {
                'query': 'How much did it rain in Tampa in May 2024? What is the rainfall on May 15?',
                'expected_value': 26.9,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Tampa, FL',
                'date': '2024-05-15'
            },
            {
                'query': 'What was the temperature in San Diego in September 2022? What is the value on September 1?',
                'expected_value': 29.4,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'San Diego, CA',
                'date': '2022-09-01'
            },

            # Midwest/Plains Queries
            {
                'query': 'What was the temperature in Kansas City in April 2023? What is the value on April 8?',
                'expected_value': 15.0,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Kansas City, MO',
                'date': '2023-04-08'
            },
            {
                'query': 'How much did it rain in Oklahoma City in March 2024? What is the rainfall on March 25?',
                'expected_value': 9.9,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Oklahoma City, OK',
                'date': '2024-03-25'
            },
            {
                'query': 'How cold was it in Milwaukee in November 2022? What is the minimum temperature on November 18?',
                'expected_value': -5.5,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Milwaukee, WI',
                'date': '2022-11-18'
            },

            # Mountain/Desert Queries
            {
                'query': 'What was the temperature in Albuquerque in May 2023? What is the value on May 10?',
                'expected_value': 27.8,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Albuquerque, NM',
                'date': '2023-05-10'
            },
            {
                'query': 'What was the snowfall in Colorado Springs in January 2024? What was the value on January 8?',
                'expected_value': 2024.0,
                'expected_unit': 'millimeters',
                'query_type': 'snow',
                'location': 'Colorado Springs, CO',
                'date': '2024-01-08'
            },

            # Northeast Queries
            {
                'query': 'What was the temperature in Philadelphia in June 2022? What is the value on June 12?',
                'expected_value': 26.7,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Philadelphia, PA',
                'date': '2022-06-12'
            },
            {
                'query': 'How much did it rain in Pittsburgh in October 2023? What is the rainfall on October 7?',
                'expected_value': 0.0,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Pittsburgh, PA',
                'date': '2023-10-07'
            },
            {
                'query': 'What was the maximum temperature in Chicago in December 2023? What is the value on December 15?',
                'expected_value': 0.3,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Chicago, IL',
                'date': '2023-12-15'
            },

            # Edge Cases - Leap Year, Month Boundaries
            {
                'query': 'What was the temperature in New York on February 2024? What is the value on February 29?',
                'expected_value': 4.2,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'New York, NY',
                'date': '2024-02-29'
            },
            {
                'query': 'what was the rainfall in Los Angeles on December 2022? What is the value on December 31?',
                'expected_value': 0.0,
                'expected_unit': 'millimeters',
                'query_type': 'precipitation',
                'location': 'Los Angeles, CA',
                'date': '2022-12-31'
            },
            {
                'query': 'Boston What was the maximum temperature in Boston on January 2024? What was the value on January 1?',
                'expected_value': 3.3,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Boston, MA',
                'date': '2024-01-01'
            },

            # Seasonal Transition Queries
            {
                'query': 'What was the maximum temperature in Denver in March 2023? What is the value on March 21?',
                'expected_value': 14.4,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Denver, CO',
                'date': '2023-03-21'
            },
            {
                'query': 'What was the maximum temperature in Minneapolis in September 2023? What is the value on September 22?',
                'expected_value': 17.8,
                'expected_unit': 'celsius',
                'query_type': 'temperature',
                'location': 'Minneapolis, MN',
                'date': '2023-09-22'
            }
        ],
        'temporal_queries': [
            {
                'query': 'Phoenix temperature July 2023',
                'expected_month': 7,
                'expected_year': 2023,
                'should_exclude_dates': ['2023-06-30', '2023-08-01']
            }
        ],
        'performance_queries': [
            'What was the temperature in Phoenix in July 2023?',
            'Compare rainfall between Seattle and Phoenix in 2022',
            'What was the hottest day in Chicago in January 2023?'
        ]
    }

class SimplePrecisionRecallF1:
    """Simple precision, recall, F1-score calculator for RAG evaluation"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def calculate_metrics(self, test_queries: List[Dict]) -> Dict[str, float]:
        """
        Calculate precision, recall, F1-score using your existing test suite

        Uses the test queries as ground truth - if the system gets the right answer,
        we consider the retrieved sources as "correct"
        """

        true_positives = 0  # Correct answers with sources retrieved
        false_positives = 0  # Wrong answers but sources retrieved
        false_negatives = 0  # Correct answers but no sources retrieved
        total_queries = len(test_queries)

        results = []

        for test_case in test_queries:
            # Get RAG response
            response = self.rag_system.query(test_case['query'])

            # Check if answer is factually correct
            retrieved_value = self._extract_numerical_value(
                response['answer'], test_case['query_type']
            )

            is_correct = self._is_within_tolerance(
                retrieved_value, test_case['expected_value'], test_case['query_type']
            )

            # Check if sources were retrieved
            sources_retrieved = len(response.get('sources', [])) > 0

            # Classify the result
            if is_correct and sources_retrieved:
                true_positives += 1
                result_type = "TP"
            elif not is_correct and sources_retrieved:
                false_positives += 1
                result_type = "FP"
            elif is_correct and not sources_retrieved:
                false_negatives += 1
                result_type = "FN"
            else:  # not correct and no sources
                # This is a true negative - system correctly didn't retrieve sources for wrong answer
                result_type = "TN"

            results.append({
                'query': test_case['query'],
                'expected': test_case['expected_value'],
                'retrieved': retrieved_value,
                'is_correct': is_correct,
                'sources_count': len(response.get('sources', [])),
                'classification': result_type
            })

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / total_queries

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_queries': total_queries,
            'detailed_results': results
        }

    def print_results(self, metrics: Dict[str, Any]):
        """Print results in a clean format"""
        print("\n" + "=" * 50)
        print("RAG SYSTEM EVALUATION RESULTS")
        print("=" * 50)
        print(f"Precision:  {metrics['precision']:.3f}")
        print(f"Recall:     {metrics['recall']:.3f}")
        print(f"F1-Score:   {metrics['f1_score']:.3f}")
        print(f"Accuracy:   {metrics['accuracy']:.3f}")
        print("\nBreakdown:")
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Total Queries:   {metrics['total_queries']}")

        # Show some example errors
        errors = [r for r in metrics['detailed_results'] if not r['is_correct']]
        if errors:
            print(f"\nSample Errors ({len(errors)} total):")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  Query: {error['query'][:60]}...")
                print(f"  Expected: {error['expected']}, Got: {error['retrieved']}")

    # Copy your existing helper methods
    def _extract_numerical_value(self, text: str, query_type: str) -> Optional[float]:
        """Extract numerical values from response text"""
        if query_type == 'temperature':
            patterns = [
                r'(-?\d+\.?\d*)\s*°C',
                r'(-?\d+\.?\d*)\s*°F',
                r'(-?\d+\.?\d*)\s*degrees?\s*(?:celsius|fahrenheit|C|F)?'
            ]
        elif query_type == 'precipitation':
            patterns = [
                r'(-?\d+\.?\d*)\s*mm',
                r'(-?\d+\.?\d*)\s*inches?',
                r'(-?\d+\.?\d*)\s*in'
            ]
        else:
            patterns = [r'(-?\d+\.?\d*)']

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    def _is_within_tolerance(self, retrieved_value: Optional[float], expected_value: float, query_type: str) -> bool:
        """Check if retrieved value is within acceptable tolerance"""
        if retrieved_value is None:
            return False

        tolerance = {
            'temperature': 0.5,  # +- 0.5
            'precipitation': 1.0,  # +- 1
            'default': 0.1  # +- 0.1
        }

        tol = tolerance.get(query_type, tolerance['default'])
        return abs(retrieved_value - expected_value) <= tol


# Simple usage - just add this to your existing code
def evaluate_with_precision_recall_f1(rag_system, test_suite):
    """Simple function to get precision, recall, F1 for your RAG system"""

    evaluator = SimplePrecisionRecallF1(rag_system)
    metrics = evaluator.calculate_metrics(test_suite['factual_queries'])
    evaluator.print_results(metrics)

    return metrics

# Example usage
if __name__ == "__main__":

    TOP_50_CITIES = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
    "Austin, TX", "Jacksonville, FL", "Fort Worth, TX", "Columbus, OH", "San Francisco, CA",
    "Charlotte, NC", "Indianapolis, IN", "Seattle, WA", "Denver, CO", "Washington, DC",
    "Boston, MA", "El Paso, TX", "Detroit, MI", "Nashville, TN", "Portland, OR",
    "Memphis, TN", "Oklahoma City, OK", "Las Vegas, NV", "Louisville, KY", "Baltimore, MD",
    "Milwaukee, WI", "Albuquerque, NM", "Tucson, AZ", "Fresno, CA", "Sacramento, CA",
    "Mesa, AZ", "Kansas City, MO", "Atlanta, GA", "Miami, FL", "Cleveland, OH",
    "New Orleans, LA", "Minneapolis, MN", "Tampa, FL", "Orlando, FL", "Pittsburgh, PA",
    "Cincinnati, OH", "St. Louis, MO", "Raleigh, NC", "Salt Lake City, UT", "Buffalo, NY"
    ]

    GEMINI_API_KEY = "your_api_key"

    # Load or initialize RAG
    rag = load_or_initialize_rag(TOP_50_CITIES, GEMINI_API_KEY, "climate_rag_50_cities")

    # System is ready - run some example queries
    print(f"\nSystem ready with {len(rag.vector_store.documents):,} documents!")

    queries = [
        "What was the temperature in Phoenix in July 2023?",
        "What's the weather forecast for New York?",
        "Show me the coldest day in Chicago in January 2023",
        "What was the hottest day in Houston during August 2024?",
        "What is the maximum temparature in New York in July 2022?",
        "Compare the difference between Maximum Temparatures in New Orleans in 2023 and Phoenix in 2022",
        "Seattle rainfall in March 2022. How much rainfall was there on March 2nd?",
        "What was the maximum temperature in Phoenix on July 2023? On which date did this occur?",
        "How hot was it in Phoenix on July 2023? What is the value on July 1?"
    ]

    print("\nRunning example queries:")
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        result = rag.query(query)
        print(result['answer'])
        print(f"Locations: {result['locations']}")
        print(f"Sources used: {len(result['sources'])}")

    #Evaluation
    # Initialize evaluator
    evaluator = ClimateRAGEvaluator(rag)

    # Create test suite
    test_suite = create_sample_test_suite()

    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(test_suite)

    # Print results
    print(json.dumps(results, indent=2, default=str))

    # Metrics
    test_suite = create_sample_test_suite()
    metrics = evaluate_with_precision_recall_f1(rag, test_suite)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")