"""
Tennis Data Loader and Processing System
Handles ATP/WTA data collection, processing, and real-time feeds
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import asyncpg
from dataclasses import asdict

# Data processing
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup
import time

from .core import PlayerProfile, MatchContext, Surface, TournamentLevel

class TennisDataLoader:
    """
    Comprehensive tennis data loading and processing system
    Integrates multiple data sources for complete tennis database
    """
    
    def __init__(self, db_path: str = "data/tennis.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Data source URLs
        self.data_sources = {
            'atp_matches': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/',
            'wta_matches': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/',
            'rankings': 'https://www.atptour.com/en/rankings/',
            'live_scores': 'https://www.flashscore.com/tennis/'
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with tennis match schema"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create comprehensive match table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                tourney_id TEXT,
                tourney_name TEXT,
                surface TEXT,
                draw_size INTEGER,
                tourney_level TEXT,
                tourney_date DATE,
                match_num INTEGER,
                
                -- Player 1 (Winner in source data)
                p1_id TEXT,
                p1_seed INTEGER,
                p1_entry TEXT,
                p1_name TEXT,
                p1_hand TEXT,
                p1_ht INTEGER,
                p1_ioc TEXT,
                p1_age REAL,
                p1_rank INTEGER,
                p1_rank_points INTEGER,
                
                -- Player 2 (Loser in source data) 
                p2_id TEXT,
                p2_seed INTEGER,
                p2_entry TEXT,
                p2_name TEXT,
                p2_hand TEXT,
                p2_ht INTEGER,
                p2_ioc TEXT,
                p2_age REAL,
                p2_rank INTEGER,
                p2_rank_points INTEGER,
                
                -- Match statistics
                score TEXT,
                best_of INTEGER,
                round TEXT,
                minutes INTEGER,
                
                -- Player 1 match stats
                p1_ace INTEGER,
                p1_df INTEGER,
                p1_svpt INTEGER,
                p1_1stIn INTEGER,
                p1_1stWon INTEGER,
                p1_2ndWon INTEGER,
                p1_SvGms INTEGER,
                p1_bpSaved INTEGER,
                p1_bpFaced INTEGER,
                
                -- Player 2 match stats
                p2_ace INTEGER,
                p2_df INTEGER,
                p2_svpt INTEGER,
                p2_1stIn INTEGER,
                p2_1stWon INTEGER,
                p2_2ndWon INTEGER,
                p2_SvGms INTEGER,
                p2_bpSaved INTEGER,
                p2_bpFaced INTEGER,
                
                -- Additional fields
                p1_won INTEGER,  -- 1 if player 1 won, 0 if player 2 won
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                hand TEXT,
                height INTEGER,
                ioc TEXT,
                birth_date DATE,
                turned_pro INTEGER,
                current_rank INTEGER,
                current_points INTEGER,
                career_wins INTEGER DEFAULT 0,
                career_losses INTEGER DEFAULT 0,
                career_titles INTEGER DEFAULT 0,
                prize_money INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create rankings history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ranking_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                ranking_date DATE,
                rank INTEGER,
                points INTEGER,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tourney_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_p1 ON matches(p1_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_p2 ON matches(p2_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_surface ON matches(surface)')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Database initialized at {self.db_path}")
    
    async def load_atp_data(self, years: List[int] = None) -> int:
        """
        Load ATP match data from Jeff Sackmann's repository
        Returns number of matches loaded
        """
        if years is None:
            years = list(range(2010, datetime.now().year + 1))
        
        total_matches = 0
        
        async with aiohttp.ClientSession() as session:
            for year in years:
                try:
                    url = f"{self.data_sources['atp_matches']}atp_matches_{year}.csv"
                    self.logger.info(f"Loading ATP data for {year}...")
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Save to temporary file and load with pandas
                            temp_file = f"temp_atp_{year}.csv"
                            with open(temp_file, 'w') as f:
                                f.write(content)
                            
                            df = pd.read_csv(temp_file)
                            matches_loaded = await self._process_and_store_matches(df, 'ATP')
                            total_matches += matches_loaded
                            
                            # Clean up temp file
                            Path(temp_file).unlink()
                            
                            self.logger.info(f"Loaded {matches_loaded} ATP matches for {year}")
                        else:
                            self.logger.warning(f"Failed to load ATP data for {year}: {response.status}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error loading ATP data for {year}: {str(e)}")
        
        self.logger.info(f"Total ATP matches loaded: {total_matches}")
        return total_matches
    
    async def load_wta_data(self, years: List[int] = None) -> int:
        """
        Load WTA match data from Jeff Sackmann's repository
        Returns number of matches loaded
        """
        if years is None:
            years = list(range(2010, datetime.now().year + 1))
        
        total_matches = 0
        
        async with aiohttp.ClientSession() as session:
            for year in years:
                try:
                    url = f"{self.data_sources['wta_matches']}wta_matches_{year}.csv"
                    self.logger.info(f"Loading WTA data for {year}...")
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Save and process
                            temp_file = f"temp_wta_{year}.csv"
                            with open(temp_file, 'w') as f:
                                f.write(content)
                            
                            df = pd.read_csv(temp_file)
                            matches_loaded = await self._process_and_store_matches(df, 'WTA')
                            total_matches += matches_loaded
                            
                            Path(temp_file).unlink()
                            
                            self.logger.info(f"Loaded {matches_loaded} WTA matches for {year}")
                        else:
                            self.logger.warning(f"Failed to load WTA data for {year}: {response.status}")
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error loading WTA data for {year}: {str(e)}")
        
        self.logger.info(f"Total WTA matches loaded: {total_matches}")
        return total_matches
    
    async def _process_and_store_matches(self, df: pd.DataFrame, tour: str) -> int:
        """
        Process and store match data in database
        Returns number of matches stored
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        matches_stored = 0
        
        for _, row in df.iterrows():
            try:
                # Create unique match ID
                match_id = f"{row.get('tourney_id', 'unknown')}_{row.get('match_num', 0)}_{tour}"
                
                # Check if match already exists
                cursor.execute("SELECT match_id FROM matches WHERE match_id = ?", (match_id,))
                if cursor.fetchone():
                    continue
                
                # Determine winner/loser (source data has winner as first player)
                # We'll randomize to create balanced training data
                if np.random.random() > 0.5:
                    # Keep original (winner first)
                    p1_won = 1
                    p1_prefix, p2_prefix = 'winner_', 'loser_'
                else:
                    # Swap players (loser first)
                    p1_won = 0
                    p1_prefix, p2_prefix = 'loser_', 'winner_'
                
                # Insert match data
                match_data = {
                    'match_id': match_id,
                    'tourney_id': row.get('tourney_id'),
                    'tourney_name': row.get('tourney_name'),
                    'surface': row.get('surface'),
                    'draw_size': row.get('draw_size'),
                    'tourney_level': row.get('tourney_level'),
                    'tourney_date': row.get('tourney_date'),
                    'match_num': row.get('match_num'),
                    'round': row.get('round'),
                    'score': row.get('score'),
                    'best_of': row.get('best_of', 3),
                    'minutes': row.get('minutes'),
                    'p1_won': p1_won,
                    
                    # Player 1 data
                    'p1_id': row.get(f'{p1_prefix}id'),
                    'p1_seed': row.get(f'{p1_prefix}seed'),
                    'p1_entry': row.get(f'{p1_prefix}entry'),
                    'p1_name': row.get(f'{p1_prefix}name'),
                    'p1_hand': row.get(f'{p1_prefix}hand'),
                    'p1_ht': row.get(f'{p1_prefix}ht'),
                    'p1_ioc': row.get(f'{p1_prefix}ioc'),
                    'p1_age': row.get(f'{p1_prefix}age'),
                    'p1_rank': row.get(f'{p1_prefix}rank'),
                    'p1_rank_points': row.get(f'{p1_prefix}rank_points'),
                    
                    # Player 2 data
                    'p2_id': row.get(f'{p2_prefix}id'),
                    'p2_seed': row.get(f'{p2_prefix}seed'),
                    'p2_entry': row.get(f'{p2_prefix}entry'),
                    'p2_name': row.get(f'{p2_prefix}name'),
                    'p2_hand': row.get(f'{p2_prefix}hand'),
                    'p2_ht': row.get(f'{p2_prefix}ht'),
                    'p2_ioc': row.get(f'{p2_prefix}ioc'),
                    'p2_age': row.get(f'{p2_prefix}age'),
                    'p2_rank': row.get(f'{p2_prefix}rank'),
                    'p2_rank_points': row.get(f'{p2_prefix}rank_points'),
                    
                    # Match statistics (swap if needed)
                    'p1_ace': row.get('w_ace' if p1_prefix == 'winner_' else 'l_ace'),
                    'p1_df': row.get('w_df' if p1_prefix == 'winner_' else 'l_df'),
                    'p1_svpt': row.get('w_svpt' if p1_prefix == 'winner_' else 'l_svpt'),
                    'p1_1stIn': row.get('w_1stIn' if p1_prefix == 'winner_' else 'l_1stIn'),
                    'p1_1stWon': row.get('w_1stWon' if p1_prefix == 'winner_' else 'l_1stWon'),
                    'p1_2ndWon': row.get('w_2ndWon' if p1_prefix == 'winner_' else 'l_2ndWon'),
                    'p1_SvGms': row.get('w_SvGms' if p1_prefix == 'winner_' else 'l_SvGms'),
                    'p1_bpSaved': row.get('w_bpSaved' if p1_prefix == 'winner_' else 'l_bpSaved'),
                    'p1_bpFaced': row.get('w_bpFaced' if p1_prefix == 'winner_' else 'l_bpFaced'),
                    
                    'p2_ace': row.get('l_ace' if p1_prefix == 'winner_' else 'w_ace'),
                    'p2_df': row.get('l_df' if p1_prefix == 'winner_' else 'w_df'),
                    'p2_svpt': row.get('l_svpt' if p1_prefix == 'winner_' else 'w_svpt'),
                    'p2_1stIn': row.get('l_1stIn' if p1_prefix == 'winner_' else 'w_1stIn'),
                    'p2_1stWon': row.get('l_1stWon' if p1_prefix == 'winner_' else 'w_1stWon'),
                    'p2_2ndWon': row.get('l_2ndWon' if p1_prefix == 'winner_' else 'w_2ndWon'),
                    'p2_SvGms': row.get('l_SvGms' if p1_prefix == 'winner_' else 'w_SvGms'),
                    'p2_bpSaved': row.get('l_bpSaved' if p1_prefix == 'winner_' else 'w_bpSaved'),
                    'p2_bpFaced': row.get('l_bpFaced' if p1_prefix == 'winner_' else 'w_bpFaced')
                }
                
                # Insert match
                placeholders = ', '.join(['?' for _ in match_data])
                columns = ', '.join(match_data.keys())
                
                cursor.execute(
                    f"INSERT OR IGNORE INTO matches ({columns}) VALUES ({placeholders})",
                    list(match_data.values())
                )
                
                matches_stored += cursor.rowcount
                
                # Update player records
                await self._update_player_record(cursor, match_data, 'p1')
                await self._update_player_record(cursor, match_data, 'p2')
                
            except Exception as e:
                self.logger.warning(f"Error processing match: {str(e)}")
                continue
        
        conn.commit()
        conn.close()
        
        return matches_stored
    
    async def _update_player_record(self, cursor, match_data: Dict, player_prefix: str):
        """Update or create player record"""
        player_id = match_data.get(f'{player_prefix}_id')
        if not player_id:
            return
        
        # Check if player exists
        cursor.execute("SELECT player_id FROM players WHERE player_id = ?", (player_id,))
        
        player_record = {
            'player_id': player_id,
            'name': match_data.get(f'{player_prefix}_name'),
            'hand': match_data.get(f'{player_prefix}_hand'),
            'height': match_data.get(f'{player_prefix}_ht'),
            'ioc': match_data.get(f'{player_prefix}_ioc')
        }
        
        if cursor.fetchone():
            # Update existing player
            cursor.execute("""
                UPDATE players SET 
                    current_rank = ?, current_points = ?, last_updated = CURRENT_TIMESTAMP
                WHERE player_id = ?
            """, (
                match_data.get(f'{player_prefix}_rank'),
                match_data.get(f'{player_prefix}_rank_points'),
                player_id
            ))
        else:
            # Insert new player
            cursor.execute("""
                INSERT INTO players (player_id, name, hand, height, ioc, current_rank, current_points)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                player_id,
                player_record['name'],
                player_record['hand'],
                player_record['height'],
                player_record['ioc'],
                match_data.get(f'{player_prefix}_rank'),
                match_data.get(f'{player_prefix}_rank_points')
            ))
    
    async def get_training_data(self, 
                              min_date: str = "2010-01-01",
                              max_date: str = None,
                              surfaces: List[str] = None) -> pd.DataFrame:
        """
        Get processed training data from database
        """
        if max_date is None:
            max_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM matches 
            WHERE tourney_date >= ? AND tourney_date <= ?
              AND p1_rank IS NOT NULL AND p2_rank IS NOT NULL
              AND p1_age IS NOT NULL AND p2_age IS NOT NULL
        """
        params = [min_date, max_date]
        
        if surfaces:
            surface_placeholders = ', '.join(['?' for _ in surfaces])
            query += f" AND surface IN ({surface_placeholders})"
            params.extend(surfaces)
        
        query += " ORDER BY tourney_date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        self.logger.info(f"Retrieved {len(df)} training matches")
        return df
    
    async def get_player_profile(self, player_name: str) -> Optional[PlayerProfile]:
        """
        Get comprehensive player profile from database
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get basic player info
        player_query = """
            SELECT * FROM players WHERE name LIKE ? ORDER BY last_updated DESC LIMIT 1
        """
        player_df = pd.read_sql_query(player_query, conn, params=[f"%{player_name}%"])
        
        if player_df.empty:
            conn.close()
            return None
        
        player_row = player_df.iloc[0]
        
        # Get career statistics
        career_stats_query = """
            SELECT 
                COUNT(*) as total_matches,
                SUM(CASE WHEN (p1_name = ? AND p1_won = 1) OR (p2_name = ? AND p1_won = 0) THEN 1 ELSE 0 END) as wins,
                COUNT(DISTINCT CASE WHEN (p1_name = ? OR p2_name = ?) AND round IN ('F', 'Final') AND 
                      ((p1_name = ? AND p1_won = 1) OR (p2_name = ? AND p1_won = 0)) THEN tourney_id END) as titles
            FROM matches 
            WHERE p1_name = ? OR p2_name = ?
        """
        
        career_df = pd.read_sql_query(career_stats_query, conn, params=[
            player_name, player_name, player_name, player_name, 
            player_name, player_name, player_name, player_name
        ])
        
        career_stats = career_df.iloc[0] if not career_df.empty else {'total_matches': 0, 'wins': 0, 'titles': 0}
        
        # Get surface-specific statistics
        surface_stats = {}
        for surface in ['Clay', 'Grass', 'Hard']:
            surface_query = """
                SELECT 
                    COUNT(*) as matches,
                    SUM(CASE WHEN (p1_name = ? AND p1_won = 1) OR (p2_name = ? AND p1_won = 0) THEN 1 ELSE 0 END) as wins
                FROM matches 
                WHERE (p1_name = ? OR p2_name = ?) AND surface = ?
            """
            surface_df = pd.read_sql_query(surface_query, conn, params=[
                player_name, player_name, player_name, player_name, surface
            ])
            
            if not surface_df.empty:
                surface_stats[surface.lower()] = {
                    'wins': int(surface_df.iloc[0]['wins']),
                    'losses': int(surface_df.iloc[0]['matches'] - surface_df.iloc[0]['wins'])
                }
        
        conn.close()
        
        # Create PlayerProfile
        profile = PlayerProfile(
            name=player_row['name'],
            player_id=player_row['player_id'],
            current_ranking=int(player_row.get('current_rank', 999)),
            current_points=int(player_row.get('current_points', 0)),
            age=float(self._calculate_age(player_row.get('birth_date'))),
            height=float(player_row.get('height', 180)),
            weight=float(75),  # Default weight
            handed=player_row.get('hand', 'R'),
            backhand='two',  # Default
            turned_pro=int(player_row.get('turned_pro', 2010)),
            career_wins=int(career_stats['wins']),
            career_losses=int(career_stats['total_matches'] - career_stats['wins']),
            career_titles=int(career_stats['titles']),
            
            # Surface stats
            clay_wins=surface_stats.get('clay', {}).get('wins', 0),
            clay_losses=surface_stats.get('clay', {}).get('losses', 0),
            grass_wins=surface_stats.get('grass', {}).get('wins', 0), 
            grass_losses=surface_stats.get('grass', {}).get('losses', 0),
            hard_wins=surface_stats.get('hard', {}).get('wins', 0),
            hard_losses=surface_stats.get('hard', {}).get('losses', 0),
        )
        
        return profile
    
    def _calculate_age(self, birth_date: Optional[str]) -> float:
        """Calculate current age from birth date"""
        if not birth_date:
            return 25.0  # Default age
        
        try:
            birth = datetime.strptime(birth_date, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            return float(age)
        except:
            return 25.0
    
    async def get_head_to_head(self, player1_id: str, player2_id: str) -> List[Dict]:
        """
        Get head-to-head match history between two players
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT tourney_date, surface, tourney_level, round, score,
                   CASE WHEN (p1_id = ? AND p1_won = 1) OR (p2_id = ? AND p1_won = 0) THEN 'p1' ELSE 'p2' END as winner
            FROM matches 
            WHERE (p1_id = ? AND p2_id = ?) OR (p1_id = ? AND p2_id = ?)
            ORDER BY tourney_date DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[
            player1_id, player1_id, player1_id, player2_id, player2_id, player1_id
        ])
        conn.close()
        
        return df.to_dict('records')
    
    async def get_recent_matches(self, player_id: str, limit: int = 20) -> List[Dict]:
        """
        Get recent matches for a player
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT tourney_date, surface, tourney_level, round, 
                   CASE WHEN (p1_id = ? AND p1_won = 1) OR (p2_id = ? AND p1_won = 0) THEN 'win' ELSE 'loss' END as result,
                   CASE WHEN p1_id = ? THEN p2_rank ELSE p1_rank END as opponent_rank,
                   score
            FROM matches 
            WHERE p1_id = ? OR p2_id = ?
            ORDER BY tourney_date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=[
            player_id, player_id, player_id, player_id, player_id, limit
        ])
        conn.close()
        
        return df.to_dict('records')
    
    async def load_current_rankings(self) -> pd.DataFrame:
        """
        Load current ATP/WTA rankings
        In production, this would scrape from official sources
        """
        # Mock current rankings data
        current_rankings = {
            'player_id': [f'player_{i}' for i in range(1, 101)],
            'name': [f'Player {i}' for i in range(1, 101)],
            'rank': list(range(1, 101)),
            'points': [8000 - i*50 for i in range(100)],
            'last_updated': [datetime.now().strftime('%Y-%m-%d')] * 100
        }
        
        return pd.DataFrame(current_rankings)
    
    async def get_database_stats(self) -> Dict:
        """
        Get database statistics and health metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Total matches
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        stats['total_matches'] = cursor.fetchone()[0]
        
        # Matches by surface
        cursor.execute("SELECT surface, COUNT(*) FROM matches GROUP BY surface")
        stats['matches_by_surface'] = dict(cursor.fetchall())
        
        # Date range
        cursor.execute("SELECT MIN(tourney_date), MAX(tourney_date) FROM matches")
        date_range = cursor.fetchone()
        stats['date_range'] = {'earliest': date_range[0], 'latest': date_range[1]}
        
        # Total players
        cursor.execute("SELECT COUNT(*) FROM players")
        stats['total_players'] = cursor.fetchone()[0]
        
        # Recent activity
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM matches WHERE tourney_date >= ?", (thirty_days_ago,))
        stats['recent_matches_30d'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats
    
    async def load_sample_data_for_testing(self) -> int:
        """
        Load sample data for testing and development
        Creates realistic match data for algorithm testing
        """
        self.logger.info("Loading sample data for testing...")
        
        # Create sample tournament data
        tournaments = [
            {'id': 'aus_open_2024', 'name': 'Australian Open', 'level': 'Grand Slam', 'surface': 'Hard'},
            {'id': 'french_open_2024', 'name': 'Roland Garros', 'level': 'Grand Slam', 'surface': 'Clay'},
            {'id': 'wimbledon_2024', 'name': 'Wimbledon', 'level': 'Grand Slam', 'surface': 'Grass'},
            {'id': 'us_open_2024', 'name': 'US Open', 'level': 'Grand Slam', 'surface': 'Hard'},
            {'id': 'miami_2024', 'name': 'Miami Open', 'level': 'Masters', 'surface': 'Hard'},
            {'id': 'monte_carlo_2024', 'name': 'Monte-Carlo Masters', 'level': 'Masters', 'surface': 'Clay'}
        ]
        
        # Create sample players (top 50)
        sample_players = [
            {'id': 'djokovic_n', 'name': 'Novak Djokovic', 'rank': 1, 'points': 11540, 'age': 37, 'hand': 'R', 'ht': 188},
            {'id': 'alcaraz_c', 'name': 'Carlos Alcaraz', 'rank': 2, 'points': 8120, 'age': 22, 'hand': 'R', 'ht': 185},
            {'id': 'sinner_j', 'name': 'Jannik Sinner', 'rank': 3, 'points': 7500, 'age': 23, 'hand': 'R', 'ht': 188},
            {'id': 'medvedev_d', 'name': 'Daniil Medvedev', 'rank': 4, 'points': 6930, 'age': 28, 'hand': 'R', 'ht': 198},
            {'id': 'zverev_a', 'name': 'Alexander Zverev', 'rank': 5, 'points': 6175, 'age': 27, 'hand': 'R', 'ht': 198},
            {'id': 'rublev_a', 'name': 'Andrey Rublev', 'rank': 6, 'points': 4805, 'age': 27, 'hand': 'R', 'ht': 188},
            {'id': 'ruud_c', 'name': 'Casper Ruud', 'rank': 7, 'points': 4630, 'age': 25, 'hand': 'R', 'ht': 183},
            {'id': 'fritz_t', 'name': 'Taylor Fritz', 'rank': 8, 'points': 4300, 'age': 27, 'hand': 'R', 'ht': 196}
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert sample players
        for player in sample_players:
            cursor.execute("""
                INSERT OR REPLACE INTO players 
                (player_id, name, hand, height, current_rank, current_points, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                player['id'], player['name'], player['hand'], player['ht'],
                player['rank'], player['points']
            ))
        
        # Generate sample matches
        matches_created = 0
        
        for tournament in tournaments:
            for round_name in ['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F']:
                # Create matches for this round
                num_matches = min(len(sample_players) // 2, 8)
                
                for i in range(num_matches):
                    player1 = sample_players[i*2 % len(sample_players)]
                    player2 = sample_players[(i*2 + 1) % len(sample_players)]
                    
                    # Determine winner based on ranking (better player more likely to win)
                    rank_diff = player1['rank'] - player2['rank']
                    p1_win_prob = 1 / (1 + 10**(rank_diff / 200))  # ELO-style calculation
                    p1_won = 1 if np.random.random() < p1_win_prob else 0
                    
                    match_id = f"{tournament['id']}_{round_name}_{i}"
                    
                    # Generate realistic match stats
                    match_data = self._generate_realistic_match_stats(
                        player1, player2, p1_won, tournament, round_name
                    )
                    match_data['match_id'] = match_id
                    
                    # Insert match
                    columns = ', '.join(match_data.keys())
                    placeholders = ', '.join(['?' for _ in match_data])
                    
                    cursor.execute(
                        f"INSERT OR IGNORE INTO matches ({columns}) VALUES ({placeholders})",
                        list(match_data.values())
                    )
                    
                    matches_created += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created {matches_created} sample matches for testing")
        return matches_created
    
    def _generate_realistic_match_stats(self, player1: Dict, player2: Dict, 
                                      p1_won: int, tournament: Dict, round_name: str) -> Dict:
        """Generate realistic match statistics"""
        # Base match data
        match_data = {
            'tourney_id': tournament['id'],
            'tourney_name': tournament['name'],
            'surface': tournament['surface'],
            'tourney_level': tournament['level'],
            'tourney_date': '2024-09-15',  # Mock date
            'round': round_name,
            'best_of': 5 if tournament['level'] == 'Grand Slam' else 3,
            'p1_won': p1_won,
            
            # Player info
            'p1_id': player1['id'],
            'p1_name': player1['name'],
            'p1_hand': player1['hand'],
            'p1_ht': player1['ht'],
            'p1_age': player1['age'],
            'p1_rank': player1['rank'],
            'p1_rank_points': player1['points'],
            
            'p2_id': player2['id'],
            'p2_name': player2['name'],
            'p2_hand': player2['hand'],
            'p2_ht': player2['ht'],
            'p2_age': player2['age'],
            'p2_rank': player2['rank'],
            'p2_rank_points': player2['points']
        }
        
        # Generate match statistics
        if p1_won:
            # Player 1 won - give them better stats
            match_data.update({
                'p1_ace': np.random.poisson(8),
                'p1_df': np.random.poisson(3),
                'p1_svpt': np.random.poisson(60),
                'p1_1stIn': np.random.poisson(40),
                'p1_1stWon': np.random.poisson(30),
                'p1_2ndWon': np.random.poisson(12),
                'p1_bpSaved': np.random.poisson(4),
                'p1_bpFaced': np.random.poisson(6),
                
                'p2_ace': np.random.poisson(5),
                'p2_df': np.random.poisson(4),
                'p2_svpt': np.random.poisson(55),
                'p2_1stIn': np.random.poisson(35),
                'p2_1stWon': np.random.poisson(25),
                'p2_2ndWon': np.random.poisson(10),
                'p2_bpSaved': np.random.poisson(2),
                'p2_bpFaced': np.random.poisson(8)
            })
            match_data['score'] = '6-4 6-2' if match_data['best_of'] == 3 else '6-4 6-2 6-3'
        else:
            # Player 2 won - give them better stats
            match_data.update({
                'p1_ace': np.random.poisson(5),
                'p1_df': np.random.poisson(4),
                'p1_svpt': np.random.poisson(55),
                'p1_1stIn': np.random.poisson(35),
                'p1_1stWon': np.random.poisson(25),
                'p1_2ndWon': np.random.poisson(10),
                'p1_bpSaved': np.random.poisson(2),
                'p1_bpFaced': np.random.poisson(8),
                
                'p2_ace': np.random.poisson(8),
                'p2_df': np.random.poisson(3),
                'p2_svpt': np.random.poisson(60),
                'p2_1stIn': np.random.poisson(40),
                'p2_1stWon': np.random.poisson(30),
                'p2_2ndWon': np.random.poisson(12),
                'p2_bpSaved': np.random.poisson(4),
                'p2_bpFaced': np.random.poisson(6)
            })
            match_data['score'] = '4-6 2-6' if match_data['best_of'] == 3 else '4-6 2-6 3-6'
        
        return match_data
    
    async def export_training_data(self, output_file: str = "data/training_data.csv") -> str:
        """
        Export processed training data for model training
        """
        training_df = await self.get_training_data()
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        training_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Training data exported to {output_file}")
        return output_file