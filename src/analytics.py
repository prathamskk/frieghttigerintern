import pandas as pd
import numpy as np
from typing import Tuple, Union, Any
from src.config import FreightConfig, DEFAULT_CONFIG

def load_and_preprocess_shipments(source: Union[str, pd.DataFrame, Any]) -> pd.DataFrame:
    """
    Loads shipment records and computes basic fields:
    - route: origin-destination
    - week_of: Monday date of the calendar week (Monday to Sunday)
    - tonne_km: quantity_tonnes * distance_km

    Accepts a filepath string, a Path-like object, an IO buffer, or an existing DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    
    # Monday of the week
    df['week_of'] = df['shipment_date'].apply(
        lambda d: (d - pd.Timedelta(days=d.weekday())).strftime('%Y-%m-%d')
    )
    df['tonne_km'] = df['quantity_tonnes'] * df['distance_km']
    df['route'] = df['origin'] + '-' + df['destination']
    return df

def compute_weekly_route_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates shipments by route, route_type, and week_of:
    cost_per_tonne_km = total freight cost / (quantity in tonnes * distance in km)
    """
    weekly = df.groupby(['route', 'origin', 'destination', 'route_type', 'week_of']).agg(
        total_cost=('freight_cost_inr', 'sum'),
        total_tonne_km=('tonne_km', 'sum'),
        shipment_count=('shipment_id', 'count')
    ).reset_index()
    
    # total_tonne_km = total_tonnes * distance_km (metric for freight industry)
    weekly['cost_per_tonne_km'] = weekly['total_cost'] / weekly['total_tonne_km']
    weekly = weekly.sort_values(['route', 'week_of']).reset_index(drop=True)
    return weekly

def compute_baselines_and_flags(
    weekly_df: pd.DataFrame,
    config: FreightConfig = DEFAULT_CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes precisely defined comparison baselines:
    1. vs. own history: trailing unweighted rolling average of cost per tonne-km
       strictly before current week (window defined by config.trailing_window_weeks).
       If fewer weeks exist, uses all available prior weeks.
    2. vs. similar routes: average cost per tonne-km in that same week across all other
       routes sharing the same route_type (excluding current route).

    Flags candidate spikes where:
       diff_hist_pct > 0 AND (diff_hist_pct >= config.spike_threshold_pct OR diff_peer_pct >= config.spike_threshold_pct)

    Returns:
    - all_weekly_df: complete table with all calculated baselines and metrics.
    - spike_candidates_df: filtered candidate spikes requiring note verification.
    """
    spike_threshold_pct = config.spike_threshold_pct
    window_weeks = config.trailing_window_weeks
    records = []
    
    for (route, rtype), group in weekly_df.groupby(['route', 'route_type']):
        group = group.sort_values('week_of').reset_index(drop=True)
        
        for idx, row in group.iterrows():
            cur_cost = row['cost_per_tonne_km']
            cur_week = row['week_of']
            
            # 1. Trailing history strictly before current week
            prior_weeks = group.iloc[max(0, idx - window_weeks):idx]
            if len(prior_weeks) > 0:
                hist_avg = prior_weeks['cost_per_tonne_km'].mean()
                diff_hist_pct = ((cur_cost - hist_avg) / hist_avg) * 100.0
                prior_week_count = len(prior_weeks)
            else:
                hist_avg = np.nan
                diff_hist_pct = np.nan
                prior_week_count = 0
                
            # 2. Peer routes in the same week with same route_type, excluding self
            peers = weekly_df[
                (weekly_df['week_of'] == cur_week) & 
                (weekly_df['route_type'] == rtype) & 
                (weekly_df['route'] != route)
            ]
            if len(peers) > 0:
                peer_avg = peers['cost_per_tonne_km'].mean()
                diff_peer_pct = ((cur_cost - peer_avg) / peer_avg) * 100.0
                peer_count = len(peers)
            else:
                peer_avg = np.nan
                diff_peer_pct = np.nan
                peer_count = 0

            # Formatting string outputs matching contract
            if not np.isnan(diff_hist_pct):
                sign = "+" if diff_hist_pct >= 0 else ""
                vs_own_history_str = f"{sign}{diff_hist_pct:.1f}% vs this route's past average"
            else:
                vs_own_history_str = "No prior history available"
                
            if not np.isnan(diff_peer_pct):
                sign = "+" if diff_peer_pct >= 0 else ""
                vs_similar_routes_str = f"{sign}{diff_peer_pct:.1f}% vs similar-length routes this week"
            else:
                vs_similar_routes_str = "No peer routes available this week"

            # Check if this row is an out-of-the-ordinary cost rise
            is_candidate_spike = False
            if not np.isnan(diff_hist_pct) and diff_hist_pct > 0:
                if diff_hist_pct >= spike_threshold_pct:
                    is_candidate_spike = True
                elif not np.isnan(diff_peer_pct) and diff_peer_pct >= spike_threshold_pct:
                    is_candidate_spike = True
                    
            records.append({
                'route': route,
                'origin': row['origin'],
                'destination': row['destination'],
                'route_type': rtype,
                'week_of': cur_week,
                'cost_per_tonne_km': round(cur_cost, 2),
                'cost_per_tonne_km_raw': cur_cost,
                'hist_avg': hist_avg,
                'diff_hist_pct': diff_hist_pct,
                'vs_own_history': vs_own_history_str,
                'peer_avg': peer_avg,
                'diff_peer_pct': diff_peer_pct,
                'vs_similar_routes': vs_similar_routes_str,
                'prior_weeks_used': prior_week_count,
                'peers_used': peer_count,
                'is_candidate_spike': is_candidate_spike
            })
            
    all_df = pd.DataFrame(records)
    candidate_spikes = all_df[all_df['is_candidate_spike']].copy().reset_index(drop=True)
    return all_df, candidate_spikes
