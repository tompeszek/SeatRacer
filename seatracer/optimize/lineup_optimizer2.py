from dataclasses import dataclass
from typing import List, Dict   


@dataclass
class LineupOptimizer2:
    """
    Class for optimizing boat lineups based on an existing Analysis model.
    
    This class takes an existing Analysis object that has been run and uses its
    model to find optimal lineups for different boat classes.
    """
    analysis: object
    available_athletes: List[str] = None
    
    def __post_init__(self):
        """Initialize the optimizer with default values if needed."""
        if self.analysis.final_results is None:
            raise ValueError("Analysis must be run before creating a LineupOptimizer")
            
        # If no athletes list provided, use all athletes from the model
        if self.available_athletes is None:
            self.available_athletes = self._get_all_athletes_from_model()
            
        # Cache athlete coefficients for faster access
        self.athlete_coefficients = self._get_athlete_coefficients()
        
        # Pre-filter athletes by their type and ranking
        self.port_athletes = self._filter_and_rank_athletes('p')
        self.starboard_athletes = self._filter_and_rank_athletes('s')
        self.sculling_athletes = self._filter_and_rank_athletes('x')

        counts = self._get_boat_positions()
        # self.mid_athletes = self._get_boat_positions('mid')
        # self.stroke_athletes = self._get_boat_positions('stroke')
    
    def _get_all_athletes_from_model(self) -> List[str]:
        """Get all athletes from the analysis model."""
        # Extract all athlete names from the model parameters
        params = self.analysis.final_results['results'].params
        
        # Athlete names typically have a suffix (ᵖ, ˢ, ˣ)
        # Note: Excluding coxswains (ᶜ) as per requirements
        suffixes = ['ᵖ', 'ˢ', 'ˣ']
        
        # Find all parameters that end with one of the suffixes
        athletes = [param for param in params.index if any(param.endswith(suffix) for suffix in suffixes)]
        
        return athletes
        
    def _get_athlete_coefficients(self) -> Dict[str, float]:
        """Get coefficients for all athletes from the model."""
        params = self.analysis.final_results['results'].params
        return {athlete: params[athlete] for athlete in self.available_athletes if athlete in params.index}
    
    def _get_boat_positions(self) -> Dict[str, float]:
        """Determine which athletes can sit where in the bow: stroke/bow.
        Bow is bow pair in 8, otherwise seat 1"""
        raw = self.analysis.final_results['raw']
        
        # Initialize tracking dictionary for each athlete
        athlete_stats = {}
        
        # Process each row in the raw dataframe
        for _, row in raw.iterrows():
            athletes = row['Personnel'].split('/')
            
            # Count appearances for each athlete
            for athlete in athletes:
                if athlete not in athlete_stats:
                    athlete_stats[athlete] = {
                        'total': 0,
                        'stroke': 0,
                        'bow': 0
                    }
                athlete_stats[athlete]['total'] += 1
            
            # Skip 2-person boats as specified
            if row['athlete_count'] == 2:
                continue

            # Stroke always counts in other boats
            stroke_athlete = athletes[0]
            athlete_stats[stroke_athlete]['stroke'] += 1
            
            # Process bow for 8-person boats
            if row['athlete_count'] == 8:
                # Athletes in positions 1 and 2 are in bow                
                athlete_stats[athletes[7]]['bow'] += 1
                athlete_stats[athletes[6]]['bow'] += 1
                
            # Process bow for 4-person boats
            elif row['athlete_count'] == 4:
                # Last athlete is in bow position (position 1)
                bow_athlete = athletes[3]  # Position 1 (last in list)
                athlete_stats[bow_athlete]['bow'] += 1
        
        # Calculate percentages and format the final result
        result = {}
        for athlete, stats in athlete_stats.items():
            result[athlete] = {
                'stroke_count': stats['stroke'],
                'bow_count': stats['bow'],
                'stroke_percent': (stats['stroke'] / stats['total']) * 100 if stats['total'] > 0 else 0,
                'bow_percent': (stats['bow'] / stats['total']) * 100 if stats['total'] > 0 else 0
            }
        
        return result
            
    
    def _filter_and_rank_athletes(self, side: str) -> List[str]:
        suffix_map = {
            'p': 'ᵖ',
            's': 'ˢ',
            'x': 'ˣ'
        }
        
        suffix = suffix_map.get(side)
        if not suffix:
            raise ValueError(f"Invalid side: {side}. Must be one of: p, s, x")
        
        # Filter athletes by side
        side_athletes = [athlete for athlete in self.available_athletes if athlete.endswith(suffix)]
        
        # Sort by coefficient (lower values = faster)
        return sorted(side_athletes, key=lambda a: self.athlete_coefficients.get(a, float('inf')))