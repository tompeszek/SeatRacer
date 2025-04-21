from itertools import combinations
from dataclasses import dataclass
from typing import List, Dict
import time as time_module
from seatracer.utils.helpers import seconds_to_time


@dataclass
class LineupOptimizer:
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
    
    def _filter_and_rank_athletes(self, side: str) -> List[str]:
        """
        Filter athletes by side and rank them by coefficient (fastest first).
        
        Parameters:
        -----------
        side : str
            'p' for port, 's' for starboard, 'x' for sculling
        
        Returns:
        --------
        List[str]
            Sorted list of athletes who row on the specified side, ranked by coefficient
        """
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
        
    def _filter_athletes_by_side(self, side: str, exclude_athletes: List[str] = None) -> List[str]:
        """
        Filter available athletes by side, accounting for exclusions.
        
        Parameters:
        -----------
        side : str
            'p' for port, 's' for starboard, 'x' for sculling
        exclude_athletes : List[str], optional
            List of athletes to exclude from consideration
        
        Returns:
        --------
        List[str]
            List of available athletes who row on the specified side
        """
        if side == 'p':
            athletes = self.port_athletes.copy()
        elif side == 's':
            athletes = self.starboard_athletes.copy()
        elif side == 'x':
            athletes = self.sculling_athletes.copy()
        else:
            raise ValueError(f"Invalid side: {side}. Must be one of: p, s, x")
        
        # Exclude athletes if specified
        if exclude_athletes:
            athletes = [a for a in athletes if a not in exclude_athletes]
            
        return athletes
    
    def _boat_class_requirements(self, boat_class: str) -> Dict[str, int]:
        """
        Determine the crew requirements for a given boat class.
        
        Parameters:
        -----------
        boat_class : str
            The boat class (e.g., '8+', '4-', '2x')
        
        Returns:
        --------
        Dict[str, int]
            Dictionary with the number of athletes needed by side ('p', 's', 'x')
        """
        # Extract number and type from boat class
        count = int(boat_class[0])
        boat_type = boat_class[1:]
        
        requirements = {
            'p': 0,
            's': 0,
            'x': 0
        }
        
        # Determine if this is a sculling or sweep boat
        if 'x' in boat_type:
            # Sculling boats have scullers only
            requirements['x'] = count
        else:
            # Sweep boats need equal port and starboard
            requirements['p'] = count // 2
            requirements['s'] = count // 2
            
        return requirements
    
    def _apply_strict_domination(self, athletes: List[str], count: int) -> List[str]:
        """
        Apply strict domination principle to reduce the search space.
        
        If an athlete is faster than at least N others (where N is the number needed),
        those slower athletes can never be in the optimal lineup together with the faster one.
        
        Parameters:
        -----------
        athletes : List[str]
            List of athletes sorted by coefficient (fastest first)
        count : int
            Number of athletes needed for the boat
            
        Returns:
        --------
        List[str]
            Filtered list of athletes that could potentially be in the optimal lineup
        """
        if len(athletes) <= count:
            # If we barely have enough athletes, return all of them
            return athletes
        
        # Quick optimization: If we need the top N athletes out of M, we only need to consider
        # 2*N-1 athletes, since if we go beyond that, we're guaranteed to have N better athletes
        max_to_consider = min(len(athletes), 2 * count - 1)
        return athletes[:max_to_consider]
    
    def _evaluate_lineup(self, personnel: List[str], boat_class: str) -> float:
        """
        Evaluate a lineup to get its predicted time.
        
        Parameters:
        -----------
        personnel : List[str]
            List of athletes in the lineup
        boat_class : str
            The boat class
            
        Returns:
        --------
        float
            Predicted time per 500m in seconds (lower is better)
        """
        try:
            return self.analysis.predict_lineup(personnel, boat_class)
        except Exception as e:
            # If prediction fails, return a very high time
            return float('inf')
    


    
    def find_optimal_lineup(self, boat_class: str, exclude_athletes: List[str] = None) -> Dict:
        """
        Find the optimal lineup for a given boat class.
        
        Parameters:
        -----------
        boat_class : str
            The boat class to optimize for
        exclude_athletes : List[str], optional
            List of athletes to exclude from consideration
            
        Returns:
        --------
        Dict
            Dictionary containing the optimal lineup information
        """
        # Start timing for performance tracking
        start_time = time_module.time()
        
        # Get boat requirements
        requirements = self._boat_class_requirements(boat_class)
        is_sculling_boat = requirements['x'] > 0
        
        # Filter athletes by side and exclude unavailable ones
        port_athletes = self._filter_athletes_by_side('p', exclude_athletes) if not is_sculling_boat else []
        starboard_athletes = self._filter_athletes_by_side('s', exclude_athletes) if not is_sculling_boat else []
        sculling_athletes = self._filter_athletes_by_side('x', exclude_athletes) if is_sculling_boat else []
        
        # Check if we have enough athletes
        if ((not is_sculling_boat and (len(port_athletes) < requirements['p'] or len(starboard_athletes) < requirements['s'])) or
            (is_sculling_boat and len(sculling_athletes) < requirements['x'])):
            return {
                'success': False,
                'error': 'Not enough athletes available for this boat class',
                'boat_class': boat_class,
                'requirements': requirements,
                'available': {
                    'port': len(port_athletes),
                    'starboard': len(starboard_athletes),
                    'sculling': len(sculling_athletes)
                }
            }
        
        # Use the strict domination principle to reduce the search space
        if not is_sculling_boat:
            port_athletes = port_athletes[:requirements['p']] #self._apply_strict_domination(port_athletes, requirements['p'])
            starboard_athletes = starboard_athletes[:requirements['s']]#self._apply_strict_domination(starboard_athletes, requirements['s'])
        else:
            sculling_athletes = sculling_athletes[:requirements['x']]#self._apply_strict_domination(sculling_athletes, requirements['x'])
        
        # Generate combinations
        best_lineup = None
        best_time = float('inf')
        
        # Function to generate combinations for a specific side
        def get_side_combinations(athletes, count):
            if count == 0:
                return [[]]
            return list(combinations(athletes, count))
        
        # Get combinations for each side
        if is_sculling_boat:
            # For sculling boats, we only need sculling athletes
            sculling_combos = get_side_combinations(sculling_athletes, requirements['x'])
            
            print(f"Evaluating {len(sculling_combos)} potential sculling lineups...")
            
            for x_combo in sculling_combos:
                # Create the lineup
                lineup = list(x_combo)
                
                # Evaluate this lineup
                time = self._evaluate_lineup(lineup, boat_class)
                
                # Update best if this is better
                if time < best_time:
                    best_time = time
                    best_lineup = lineup
        else:
            # For sweep boats, we need port and starboard athletes
            port_combos = get_side_combinations(port_athletes, requirements['p'])
            starboard_combos = get_side_combinations(starboard_athletes, requirements['s'])
            
            total_combos = len(port_combos) * len(starboard_combos)
            print(f"Evaluating {total_combos} potential sweep lineups...")
            
            count = 0
            for p_combo in port_combos:
                for s_combo in starboard_combos:
                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count}/{total_combos} lineups...")
                    
                    # Create the lineup
                    lineup = list(p_combo) + list(s_combo)
                    
                    # Evaluate this lineup
                    time = self._evaluate_lineup(lineup, boat_class)
                    
                    # Update best if this is better
                    if time < best_time:
                        best_time = time
                        best_lineup = lineup
        
        if best_lineup is None:
            return {
                'success': False,
                'error': 'Could not find a valid lineup',
                'boat_class': boat_class
            }
        
        # Format the result
        elapsed_time = time_module.time() - start_time
        return {
            'success': True,
            'boat_class': boat_class,
            'personnel': best_lineup,
            'predicted_time': best_time,
            'formatted_time': seconds_to_time(best_time),
            'computation_time': elapsed_time
        }
    
    def create_boat_sequence(self, boat_classes: List[str]) -> List[Dict]:
        """
        Create a sequence of optimal boats for multiple boat classes.
        
        This method optimizes each boat in sequence, removing athletes that 
        have been selected for previous boats.
        
        Parameters:
        -----------
        boat_classes : List[str]
            List of boat classes in priority order
            
        Returns:
        --------
        List[Dict]
            List of dictionaries containing the optimal lineup information for each boat
        """
        results = []
        used_athletes = set()
        start_time = time_module.time()
        
        for boat_class in boat_classes:
            # Time the optimization of each boat
            boat_start_time = time_module.time()
            
            result = self.find_optimal_lineup(boat_class, list(used_athletes))
            
            # Add timing information
            boat_time = time_module.time() - boat_start_time
            if result['success']:
                result['boat_computation_time'] = boat_time
                results.append(result)
                used_athletes.update(result['personnel'])
            else:
                # If we couldn't make a valid lineup, add the error result
                result['boat_computation_time'] = boat_time
                results.append(result)
        
        # Add overall timing information
        total_time = time_module.time() - start_time
        for result in results:
            result['total_computation_time'] = total_time
        
        return results
        
    def analyze_lineups(self, boat_class: str, lineups: List[List[str]]) -> List[Dict]:
        """
        Analyze a set of provided lineups to compare them.
        
        Parameters:
        -----------
        boat_class : str
            The boat class for the lineups
        lineups : List[List[str]]
            List of lineups to analyze
            
        Returns:
        --------
        List[Dict]
            List of dictionaries containing analysis information for each lineup
        """
        results = []
        
        for i, lineup in enumerate(lineups):
            # Evaluate this lineup
            time = self._evaluate_lineup(lineup, boat_class)
            
            if time != float('inf'):  # Only include valid lineups
                results.append({
                    'success': True,
                    'name': f"Lineup {i+1}",
                    'boat_class': boat_class,
                    'personnel': lineup,
                    'predicted_time': time,
                    'formatted_time': seconds_to_time(time)
                })
            else:
                results.append({
                    'success': False,
                    'name': f"Lineup {i+1}",
                    'boat_class': boat_class,
                    'personnel': lineup,
                    'error': 'Invalid lineup'
                })
        
        # Sort results by predicted time (fastest first)
        valid_results = [r for r in results if r['success']]
        invalid_results = [r for r in results if not r['success']]
        
        valid_results.sort(key=lambda x: x['predicted_time'])
        
        return valid_results + invalid_results
    
    def find_top_n_lineups(self, boat_class: str, n: int = 5) -> List[Dict]:
        """
        Find the top N lineups for a given boat class.
        
        Parameters:
        -----------
        boat_class : str
            The boat class to optimize for
        n : int, optional
            Number of lineups to return (default: 5)
            
        Returns:
        --------
        List[Dict]
            List of dictionaries containing the top N lineup information
        """
        start_time = time_module.time()
        
        # Get boat requirements
        requirements = self._boat_class_requirements(boat_class)
        is_sculling_boat = requirements['x'] > 0
        
        # Filter athletes by side
        port_athletes = self._filter_athletes_by_side('p') if not is_sculling_boat else []
        starboard_athletes = self._filter_athletes_by_side('s') if not is_sculling_boat else []
        sculling_athletes = self._filter_athletes_by_side('x') if is_sculling_boat else []
        
        # Check if we have enough athletes
        if ((not is_sculling_boat and (len(port_athletes) < requirements['p'] or len(starboard_athletes) < requirements['s'])) or
            (is_sculling_boat and len(sculling_athletes) < requirements['x'])):
            return [{
                'success': False,
                'error': 'Not enough athletes available for this boat class',
                'boat_class': boat_class
            }]
        
        # Use strict domination but with a larger buffer since we want multiple lineups
        # We'll use 3*N to ensure we have enough combinations for diversity
        if not is_sculling_boat:
            port_athletes = self._apply_strict_domination(port_athletes, requirements['p'] * 3)
            starboard_athletes = self._apply_strict_domination(starboard_athletes, requirements['s'] * 3)
        else:
            sculling_athletes = self._apply_strict_domination(sculling_athletes, requirements['x'] * 3)
        
        # Create a priority queue of top lineups
        from heapq import heappush, heappushpop, nlargest
        top_lineups = []  # Min-heap (negative time values for max-heap behavior)
        
        # Function to generate combinations for a specific side
        def get_side_combinations(athletes, count):
            if count == 0:
                return [[]]
            return list(combinations(athletes, count))
        
        # Get combinations for each side
        if is_sculling_boat:
            # For sculling boats, we only need sculling athletes
            sculling_combos = get_side_combinations(sculling_athletes, requirements['x'])
            
            print(f"Evaluating {len(sculling_combos)} potential sculling lineups...")
            
            for x_combo in sculling_combos:
                # Create the lineup
                lineup = list(x_combo)
                
                # Evaluate this lineup
                time = self._evaluate_lineup(lineup, boat_class)
                
                if time != float('inf'):  # Only include valid lineups
                    # Use negative time for min-heap to act as max-heap (we want fastest times)
                    lineup_info = {
                        'success': True,
                        'boat_class': boat_class,
                        'personnel': lineup,
                        'predicted_time': time,
                        'formatted_time': seconds_to_time(time)
                    }
                    
                    if len(top_lineups) < n:
                        heappush(top_lineups, (-time, lineup_info))
                    else:
                        # Replace the worst lineup if this one is better
                        if -time > top_lineups[0][0]:
                            heappushpop(top_lineups, (-time, lineup_info))
        else:
            # For sweep boats, we need port and starboard athletes
            port_combos = get_side_combinations(port_athletes, requirements['p'])
            starboard_combos = get_side_combinations(starboard_athletes, requirements['s'])
            
            total_combos = len(port_combos) * len(starboard_combos)
            print(f"Evaluating {total_combos} potential sweep lineups...")
            
            count = 0
            for p_combo in port_combos:
                for s_combo in starboard_combos:
                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count}/{total_combos} lineups...")
                    
                    # Create the lineup
                    lineup = list(p_combo) + list(s_combo)
                    
                    # Evaluate this lineup
                    time = self._evaluate_lineup(lineup, boat_class)
                    
                    if time != float('inf'):  # Only include valid lineups
                        # Use negative time for min-heap to act as max-heap (we want fastest times)
                        lineup_info = {
                            'success': True,
                            'boat_class': boat_class,
                            'personnel': lineup,
                            'predicted_time': time,
                            'formatted_time': seconds_to_time(time)
                        }
                        
                        if len(top_lineups) < n:
                            heappush(top_lineups, (-time, lineup_info))
                        else:
                            if -time > top_lineups[0][0]:
                                heappushpop(top_lineups, (-time, lineup_info))
        
        # Extract lineup info from heap (sorted by time)
        result_lineups = [item[1] for item in sorted(top_lineups, key=lambda x: -x[0])]
        
        elapsed_time = time_module.time() - start_time
        for lineup in result_lineups:
            lineup['computation_time'] = elapsed_time
        
        return result_lineups
    
    def find_alternative_lineups(self, optimal_lineup: Dict, max_changes: int = 2, n: int = 5) -> List[Dict]:
        """
        Find alternative lineups that are close to the optimal lineup.
        
        Parameters:
        -----------
        optimal_lineup : Dict
            The optimal lineup to use as a base
        max_changes : int, optional
            Maximum number of athletes to change (default: 2)
        n : int, optional
            Number of alternative lineups to return (default: 5)
            
        Returns:
        --------
        List[Dict]
            List of dictionaries containing alternative lineup information
        """
        start_time = time_module.time()
        
        if not optimal_lineup['success']:
            return [{'success': False, 'error': 'Optimal lineup is not valid'}]
        
        boat_class = optimal_lineup['boat_class']
        optimal_personnel = optimal_lineup['personnel']
        requirements = self._boat_class_requirements(boat_class)
        is_sculling_boat = requirements['x'] > 0
        
        # Identify athlete types in the optimal lineup
        optimal_ports = [a for a in optimal_personnel if a.endswith('ᵖ')]
        optimal_starboards = [a for a in optimal_personnel if a.endswith('ˢ')]
        optimal_scullers = [a for a in optimal_personnel if a.endswith('ˣ')]
        
        # Get available athletes not in the optimal lineup
        available_ports = [a for a in self._filter_athletes_by_side('p') if a not in optimal_ports]
        available_starboards = [a for a in self._filter_athletes_by_side('s') if a not in optimal_starboards]
        available_scullers = [a for a in self._filter_athletes_by_side('x') if a not in optimal_scullers]
        
        # Create a priority queue of top alternative lineups
        from heapq import heappush, heappushpop
        alternative_lineups = []  # Min-heap (negative time values for max-heap behavior)
        
        # Try different combinations of substitutions
        for changes in range(1, max_changes + 1):
            if is_sculling_boat:
                # For sculling boats, we only need to substitute scullers
                for x_out_count in range(1, min(changes + 1, len(optimal_scullers) + 1)):
                    # Skip if we're trying to substitute more athletes than available
                    if x_out_count > 0 and len(available_scullers) < x_out_count:
                        continue
                    
                    # Get combinations of athletes to substitute
                    x_out_combos = list(combinations(optimal_scullers, x_out_count))
                    x_in_combos = list(combinations(available_scullers, x_out_count))
                    
                    # Try all combinations of substitutions
                    for x_out in x_out_combos:
                        for x_in in x_in_combos:
                            # Create new lineup
                            new_lineup = optimal_personnel.copy()
                            
                            # Remove athletes to substitute
                            for athlete in x_out:
                                new_lineup.remove(athlete)
                            
                            # Add replacement athletes
                            new_lineup.extend(x_in)
                            
                            # Evaluate this lineup
                            time = self._evaluate_lineup(new_lineup, boat_class)
                            
                            if time != float('inf'):  # Only include valid lineups
                                # Create a substitution description
                                substitutions = [f"{athlete_out} → {athlete_in}" for athlete_out, athlete_in in zip(x_out, x_in)]
                                
                                lineup_info = {
                                    'success': True,
                                    'boat_class': boat_class,
                                    'personnel': new_lineup,
                                    'predicted_time': time,
                                    'formatted_time': seconds_to_time(time),
                                    'substitutions': substitutions,
                                    'substitution_count': len(substitutions)
                                }
                                
                                if len(alternative_lineups) < n:
                                    heappush(alternative_lineups, (-time, lineup_info))
                                else:
                                    # Replace the worst lineup if this one is better
                                    if -time > alternative_lineups[0][0]:
                                        heappushpop(alternative_lineups, (-time, lineup_info))
            else:
                # For sweep boats, we need to substitute port and starboard athletes
                for p_out_count in range(min(changes + 1, len(optimal_ports) + 1)):
                    s_out_count = min(changes - p_out_count, len(optimal_starboards))
                    
                    # Skip invalid combinations
                    if p_out_count + s_out_count == 0 or p_out_count + s_out_count > changes:
                        continue
                    
                    # Skip if we're trying to substitute more athletes than available
                    if (p_out_count > 0 and len(available_ports) < p_out_count or
                        s_out_count > 0 and len(available_starboards) < s_out_count):
                        continue
                    
                    # Get combinations of athletes to substitute
                    p_out_combos = list(combinations(optimal_ports, p_out_count)) if p_out_count > 0 else [[]]
                    s_out_combos = list(combinations(optimal_starboards, s_out_count)) if s_out_count > 0 else [[]]
                    
                    # Get combinations of replacement athletes
                    p_in_combos = list(combinations(available_ports, p_out_count)) if p_out_count > 0 else [[]]
                    s_in_combos = list(combinations(available_starboards, s_out_count)) if s_out_count > 0 else [[]]
                    
                    # Try all combinations of substitutions
                    for p_out in p_out_combos:
                        for p_in in p_in_combos:
                            for s_out in s_out_combos:
                                for s_in in s_in_combos:
                                    # Skip if any list of athletes to add is empty but we need to add
                                    if (p_out_count > 0 and not p_in) or (s_out_count > 0 and not s_in):
                                        continue
                                    
                                    # Create new lineup
                                    new_lineup = optimal_personnel.copy()
                                    
                                    # Remove athletes to substitute
                                    for athlete in p_out + s_out:
                                        new_lineup.remove(athlete)
                                    
                                    # Add replacement athletes
                                    new_lineup.extend(p_in + s_in)
                                    
                                    # Evaluate this lineup
                                    time = self._evaluate_lineup(new_lineup, boat_class)
                                    
                                    if time != float('inf'):  # Only include valid lineups
                                        # Create a substitution description
                                        substitutions = []
                                        for athlete_out, athlete_in in list(zip(p_out, p_in)) + list(zip(s_out, s_in)):
                                            substitutions.append(f"{athlete_out} → {athlete_in}")
                                        
                                        lineup_info = {
                                            'success': True,
                                            'boat_class': boat_class,
                                            'personnel': new_lineup,
                                            'predicted_time': time,
                                            'formatted_time': seconds_to_time(time),
                                            'substitutions': substitutions,
                                            'substitution_count': len(substitutions)
                                        }
                                        
                                        if len(alternative_lineups) < n:
                                            heappush(alternative_lineups, (-time, lineup_info))
                                        else:
                                            # Replace the worst lineup if this one is better
                                            if -time > alternative_lineups[0][0]:
                                                heappushpop(alternative_lineups, (-time, lineup_info))
        
        # Extract lineup info from heap (sorted by time)
        result_lineups = [item[1] for item in sorted(alternative_lineups, key=lambda x: -x[0])]
        
        elapsed_time = time_module.time() - start_time
        for lineup in result_lineups:
            lineup['computation_time'] = elapsed_time
            
        return result_lineups