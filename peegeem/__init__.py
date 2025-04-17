

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination, CausalInference # Use both
from pgmpy.factors.discrete import DiscreteFactor
import numpy as np
from typing import Union, List, Any, Dict, Tuple, Optional
import itertools
import re

# Forward declaration for type hinting
class DAG: pass
class Variable: pass
class VariablCombination: pass
class Condition: pass
class QueryExpression: pass
class TargetStateQuery: pass
class DoCondition: pass
class DoRangeCondition: pass

class Condition:
    """Represents a standard evidence condition (e.g., variable == value)."""
    def __init__(self, variable_name: str, operator: str, value: Any, dag_instance: DAG):
        self.variable_name = variable_name
        self.operator = operator
        self.value = value
        self._dag = dag_instance

    def __repr__(self): 
        return f"Condition({self.variable_name} {self.operator} {repr(self.value)})"

    def __and__(self, other: Union['Condition', 'DoCondition', 'DoRangeCondition', List[Any]]) -> List[Any]:
        valid_types = (Condition, DoCondition, DoRangeCondition)
        if isinstance(other, valid_types):
            if self._dag is not other._dag: 
                raise ValueError("Cannot combine conditions/interventions from different DAG instances.")
            return [self, other]
        elif isinstance(other, list):
            if not all((isinstance(c, valid_types) and c._dag is self._dag) for c in other): 
                raise ValueError("Invalid item or DAG mismatch in condition list.")
            return [self] + other
        else: 
            return NotImplemented

    def __rand__(self, other: Union[Any]) -> List[Any]:
         valid_types = (Condition, DoCondition, DoRangeCondition)
         if isinstance(other, list):
             if not all((isinstance(c, valid_types) and c._dag is self._dag) for c in other): 
                 raise ValueError("Invalid item or DAG mismatch in condition list.")
             return other + [self]
         return NotImplemented

class DoCondition:
    """Represents a causal intervention condition (do(variable = value))."""
    def __init__(self, variable_name: str, value: Any, dag_instance: DAG):
        self.variable_name = variable_name
        self.value = value
        self._dag = dag_instance
        self.operator = 'do=='

    def __repr__(self): 
        return f"Do({self.variable_name}={repr(self.value)})"

    def __and__(self, other: Union[Condition, 'DoCondition', 'DoRangeCondition', List[Any]]) -> List[Any]:
        valid_types = (Condition, DoCondition, DoRangeCondition)
        if isinstance(other, valid_types):
            if self._dag is not other._dag: 
                raise ValueError("Cannot combine conditions/interventions from different DAG instances.")
            return [self, other]
        elif isinstance(other, list):
            if not all((isinstance(c, valid_types) and c._dag is self._dag) for c in other): 
                raise ValueError("Invalid item or DAG mismatch in condition list.")
            return [self] + other
        else: 
            return NotImplemented

    def __rand__(self, other: Union[Any]) -> List[Any]:
         valid_types = (Condition, DoCondition, DoRangeCondition)
         if isinstance(other, list):
             if not all((isinstance(c, valid_types) and c._dag is self._dag) for c in other): 
                 raise ValueError("Invalid item or DAG mismatch in condition list.")
             return other + [self]
         return NotImplemented

class DoRangeCondition:
    """Represents a causal 'intervention' specified by a range (e.g., do(age > 40))."""
    def __init__(self, variable_name: str, operator: str, value: Any, dag_instance: DAG):
        self.variable_name = variable_name 
        self.operator = operator 
        self.value = value 
        self._dag = dag_instance

    def __repr__(self): 
        return f"DoRange({self.variable_name} {self.operator} {repr(self.value)})"

    def __and__(self, other: Union[Condition, DoCondition, 'DoRangeCondition', List[Any]]) -> List[Any]:
        valid_types = (Condition, DoCondition, DoRangeCondition)
        if isinstance(other, valid_types):
            if self._dag is not other._dag: 
                raise ValueError("Cannot combine conditions/interventions from different DAG instances.")
            return [self, other]
        elif isinstance(other, list):
            if not all((isinstance(c, valid_types) and c._dag is self._dag) for c in other): 
                raise ValueError("Invalid item or DAG mismatch in condition list.")
            return [self] + other
        else: 
            return NotImplemented

    def __rand__(self, other: Union[Any]) -> List[Any]:
         valid_types = (Condition, DoCondition, DoRangeCondition)
         if isinstance(other, list):
             if not all((isinstance(c, valid_types) and c._dag is self._dag) for c in other): 
                 raise ValueError("Invalid item or DAG mismatch in condition list.")
             return other + [self]
         return NotImplemented

def do(condition_obj: Condition) -> Union[DoCondition, DoRangeCondition]:
    """Creates a DoCondition or DoRangeCondition for causal intervention/filtering."""
    if not isinstance(condition_obj, Condition): 
        raise TypeError("Argument to do() must be a condition created from a Variable (e.g., variable == value, variable > 40).")
    if condition_obj.operator == '==':
        return DoCondition(
            variable_name=condition_obj.variable_name, 
            value=condition_obj.value, 
            dag_instance=condition_obj._dag
        )
    elif condition_obj.operator in ['!=', '>', '>=', '<', '<=']:
         return DoRangeCondition(
             variable_name=condition_obj.variable_name, 
             operator=condition_obj.operator, 
             value=condition_obj.value, 
             dag_instance=condition_obj._dag
         )
    else: raise ValueError(f"Unsupported operator '{condition_obj.operator}' passed to do().")


class VariableCombination:
    """Represents a combination of multiple variables for joint probability queries."""
    def __init__(self, variables: List[Variable]):
        if not variables: 
            raise ValueError("VariableCombination cannot be empty.")
        first_dag = variables[0]._dag
        if not all(isinstance(v, Variable) and v._dag is first_dag for v in variables): 
            raise ValueError("All items must be Variables from the same DAG.")
        self.variables = list(dict.fromkeys(variables))
        self._dag = first_dag

    def __repr__(self): 
        return f"VariableCombination([{', '.join(v.name for v in self.variables)}])"

    def __and__(self, other: Union[Variable, 'VariableCombination']) -> 'VariableCombination':
        if isinstance(other, Variable):
            if other._dag is not self._dag: 
                raise ValueError("Cannot combine variables from different DAGs.")
            return VariableCombination(self.variables + [other])
        elif isinstance(other, VariableCombination):
            if other._dag is not self._dag: 
                raise ValueError("Cannot combine variables from different DAGs.")
            return VariableCombination(self.variables + other.variables)
        return NotImplementedi

    def __rand__(self, other: Variable) -> 'VariableCombination':
         if isinstance(other, Variable):
             if other._dag is not self._dag: 
                 raise ValueError("Cannot combine variables from different DAGs.")
             return VariableCombination([other] + self.variables)
         return NotImplemented

    def __or__(self, conditions: Union[Condition, DoCondition, DoRangeCondition, List[Any]]) -> 'QueryExpression':
        return QueryExpression(target=self, conditions=conditions)


class TargetStateQuery:
    """Represents a query for the probability of a specific state of a variable."""
    def __init__(self, variable: Variable, state: Any):
        self.variable = variable
        processed_state, _ = variable._process_value_for_state_check(state)
        self.state = processed_state 
        self._dag = variable._dag

    def __repr__(self): 
        return f"TargetStateQuery({self.variable.name} == {repr(self.state)})"

    def __or__(self, conditions: Union[Condition, DoCondition, DoRangeCondition, List[Any]]) -> 'QueryExpression':
        """Creates query P(var == state | conditions/interventions)."""
        return QueryExpression(target=self, conditions=conditions)


class Variable:
    """Represents a variable (node) in the DAG with overloaded operators. Made hashable."""
    def __init__(self, name: str, dag_instance: DAG): 
        self.name = name; self._dag = dag_instance

    def __repr__(self): 
        return f"Variable({self.name})"

    def __str__(self): 
        return self.name

    def is_(self, value: Any) -> TargetStateQuery: 
        return TargetStateQuery(self, value)

    def __eq__(self, other: object) -> Union[bool, Condition]:
        if isinstance(other, Variable): 
            return self.name == other.name and self._dag is other._dag
        else: 
            processed_value, _ = self._process_value_for_state_check(other)
            return Condition(self.name, '==', processed_value, self._dag)

    def __ne__(self, other: object) -> Condition:
        if isinstance(other, Variable): 
            raise TypeError("'!=' between Variable objects not supported.")
        processed_value, _ = self._process_value_for_state_check(other)
        return Condition(self.name, '!=', processed_value, self._dag)

    def __gt__(self, other: object) -> Condition:
        if isinstance(other, Variable): 
            raise TypeError("'>' between Variable objects not supported.")
        processed_value, is_numeric = self._process_value_for_state_check(other)

        if not is_numeric: 
            print(f"Warning: Applying '>' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
        return Condition(self.name, '>', processed_value, self._dag)

    def __ge__(self, other: object) -> Condition:
        if isinstance(other, Variable): 
            raise TypeError("'>=' between Variable objects not supported.")
        processed_value, is_numeric = self._process_value_for_state_check(other)
        if not is_numeric: 
            print(f"Warning: Applying '>=' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
        return Condition(self.name, '>=', processed_value, self._dag)

    def __lt__(self, other: object) -> Condition:
        if isinstance(other, Variable): 
            raise TypeError("'<' between Variable objects not supported.")
        processed_value, is_numeric = self._process_value_for_state_check(other)
        if not is_numeric: 
            print(f"Warning: Applying '<' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
        return Condition(self.name, '<', processed_value, self._dag)

    def __le__(self, other: object) -> Condition:
        if isinstance(other, Variable): 
            raise TypeError("'<=' between Variable objects not supported.")
        processed_value, is_numeric = self._process_value_for_state_check(other)
        if not is_numeric: 
            print(f"Warning: Applying '<=' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
        return Condition(self.name, '<=', processed_value, self._dag)

    def _process_value_for_state_check(self, value: Any) -> Tuple[Any, bool]:
        is_numeric = isinstance(value, (int, float))
        try:
            if not hasattr(self._dag, 'model') or not hasattr(self._dag, '_bool_cols'): raise AttributeError("DAG model not fully initialized.")
            processed_value = str(value) if self.name in self._dag._bool_cols else value
            return processed_value, is_numeric
        except Exception as e: 
            print(f"Warning: Could not fully process/verify value '{repr(value)}' for var '{self.name}': {e}")
            return value, is_numeric

    def __or__(self, conditions: Union[Condition, DoCondition, DoRangeCondition, List[Any]]) -> 'QueryExpression':
        """Creates a query expression using '|' for 'given': target | conditions/interventions."""
        return QueryExpression(target=self, conditions=conditions)

    def __and__(self, other: Union[Variable, VariableCombination]) -> VariableCombination:
        if isinstance(other, Variable):
            if other._dag is not self._dag: 
                raise ValueError("Cannot combine variables from different DAGs.")
            return VariableCombination([self, other])
        elif isinstance(other, VariableCombination): 
            return other.__rand__(self)
        return NotImplemented

    def __hash__(self) -> int: 
        return hash(self.name)


class QueryExpression:
    """Represents a probability query expression with potential interventions."""
    def __init__(self, target: Union[Variable, VariableCombination, TargetStateQuery],
                 conditions: Union[Condition, DoCondition, DoRangeCondition, List[Any]]): # Accept list of Any and validate below
        self.target = target
        self._dag = target._dag
        valid_cond_types = (Condition, DoCondition, DoRangeCondition)
        if isinstance(conditions, valid_cond_types): self.conditions = [conditions]
        elif isinstance(conditions, list): self.conditions = conditions
        else: raise TypeError(f"Conditions must be Condition, DoCondition, DoRangeCondition or list thereof, got {type(conditions)}")
        if not all(isinstance(c, valid_cond_types) and c._dag is self._dag for c in self.conditions):
             raise ValueError("All conditions/interventions must be valid types and belong to the same DAG as the target.")
    @property
    def target_names(self) -> List[str]:
        if isinstance(self.target, Variable): return [self.target.name]
        elif isinstance(self.target, VariableCombination): return [v.name for v in self.target.variables]
        elif isinstance(self.target, TargetStateQuery): return [self.target.variable.name]
        else: raise TypeError("Internal Error: Invalid target type in QueryExpression")
    def __repr__(self): return f"QueryExpression(Target: {repr(self.target)}, Conditions: {self.conditions})"


def _get_matching_states(variable_name: str, operator: str, value: Any, dag_instance: DAG) -> List[Any]:
    """Identifies discrete states of a variable that satisfy a given condition."""
    try: 
        cpd = dag_instance.model.get_cpds(variable_name)
        all_states = cpd.state_names[variable_name] if cpd else []
    except Exception as e: 
        print(f"Warning: Could not retrieve states for variable '{variable_name}': {e}") 
        return []
    matching_states = []
    value_is_numeric = isinstance(value, (int, float))

    for state in all_states:
        state_matches = False
        try:
            state_numeric = None
            range_match = re.match(r"([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)", str(state))
            try: 
                state_numeric = float(state)
                is_single_number_state = True
            except ValueError: 
                is_single_number_state = False
            if value_is_numeric:
                if range_match:
                    parsed_lower = float(range_match.group(1))
                    is_plus_range = range_match.group(2) == '+'
                    parsed_upper = float('inf') if is_plus_range else float(range_match.group(2))
                    if operator == '>': state_matches = parsed_lower > value
                    elif operator == '>=': state_matches = parsed_lower >= value
                    elif operator == '<': state_matches = parsed_upper < value
                    elif operator == '<=': state_matches = parsed_upper <= value
                    elif operator == '==': state_matches = parsed_lower <= value < parsed_upper
                    elif operator == '!=': state_matches = not (parsed_lower <= value < parsed_upper)
                elif is_single_number_state:
                    if operator == '>': state_matches = state_numeric > value
                    elif operator == '>=': state_matches = state_numeric >= value
                    elif operator == '<': state_matches = state_numeric < value
                    elif operator == '<=': state_matches = state_numeric <= value
                    elif operator == '==': state_matches = np.isclose(state_numeric, value)
                    elif operator == '!=': state_matches = not np.isclose(state_numeric, value)
                else:
                     if isinstance(value, str):
                         if operator == '==': state_matches = (state == value)
                         elif operator == '!=': state_matches = (state != value)
            else:
                 if operator == '==': state_matches = (str(state) == str(value))
                 elif operator == '!=': state_matches = (str(state) != str(value))
        except Exception as parse_error: print(f"Warning: Could not parse state '{state}': {parse_error}"); state_matches = False
        if state_matches: matching_states.append(state)
    if not matching_states: 
        print(f"Warning: Condition '{variable_name} {operator} {repr(value)}' did not match any discrete states.")
    return matching_states

# --- Updated P_Calculator Class (Standalone) ---
class P_Calculator:
    """
    Callable class to calculate probabilities. Handles interventions via do().
    Handles marginal, joint, conditional, and specific state queries.
    Handles equality and range conditions (observational and interventional via filtering).
    Returns List[Dict] for distributions, float for specific state queries.
    Includes method to generate LaTeX representation of queries.
    """

    def _latex_escape(self, text: str) -> str:
        """Basic LaTeX escaping for variable names and string values."""
        # More robust escaping might be needed for edge cases
        return text.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')\
                   .replace('#', '\\#').replace('$', '\\$')
                   #.replace('{', '\\{').replace('}', '\\}') # Avoid escaping braces used by \text

    def _format_latex_condition(self, cond: Union[Condition, DoCondition, DoRangeCondition]) -> str:
        """Formats a single condition object into a LaTeX string."""
        var = self._latex_escape(str(cond.variable_name))
        op = cond.operator
        val = cond.value

        # Wrap variable name in \text{}
        latex_var = f"\\text{{{var}}}"

        # Format value
        if isinstance(val, str):
             val_repr = f"\\texttt{{{self._latex_escape(val)}}}"
        elif isinstance(val, bool):
             val_repr = "\\texttt{True}" if val else "\\texttt{False}"
        else:
             val_repr = str(val) # Keep numbers as is

        # Format operator
        latex_op = op.replace('>=', '\\ge ').replace('<=', '\\le ').replace('!=', '\\neq ')
        if latex_op == '==': latex_op = '=' # Use '=' in LaTeX for equality condition

        if isinstance(cond, DoCondition):
            return f"do({latex_var}={val_repr})"
        elif isinstance(cond, DoRangeCondition):
            return f"do({latex_var}{latex_op}{val_repr})"
        elif isinstance(cond, Condition):
            return f"{latex_var}{latex_op}{val_repr}"
        else:
            return ""

    def to_latex(self, query_input: Union[Variable, VariableCombination, TargetStateQuery, QueryExpression]) -> str:
        """
        Generates a LaTeX string representation of the probability query.

        Args:
            query_input: Variable, VariableCombination, TargetStateQuery, or QueryExpression.

        Returns:
            str: A LaTeX string representing the query (e.g., "P(\\text{outcome} \\mid \\text{smoking}=\\texttt{True})").
        """
        target_obj: Optional[Union[Variable, VariableCombination, TargetStateQuery]] = None
        all_conditions: List[Union[Condition, DoCondition, DoRangeCondition]] = []

        # Parse input to find target and conditions
        if isinstance(query_input, (Variable, VariableCombination)): 
            target_obj = query_input
        elif isinstance(query_input, TargetStateQuery): 
            target_obj = query_input
        elif isinstance(query_input, QueryExpression):
            target_obj = query_input.target; all_conditions = query_input.conditions
        else: 
            raise TypeError("Input must be a Variable, VariableCombination, TargetStateQuery, or QueryExpression.")

        if target_obj is None: 
            raise ValueError("Query target is missing.")

        # Format target part
        target_str = ""
        if isinstance(target_obj, Variable):
            # Wrap variable name in \text
            target_str = f"\\text{{{self._latex_escape(str(target_obj.name))}}}"
        elif isinstance(target_obj, VariableCombination):
            # Wrap each variable name in \text
            target_str = ", ".join(f"\\text{{{self._latex_escape(str(v.name))}}}" for v in target_obj.variables)
        elif isinstance(target_obj, TargetStateQuery):
            # Format as Var = Val, wrap var name in \text
            var_name = f"\\text{{{self._latex_escape(str(target_obj.variable.name))}}}"
            state_val = target_obj.state
            if isinstance(state_val, str):
                 state_repr = f"\\texttt{{{self._latex_escape(state_val)}}}"
            elif isinstance(state_val, bool):
                 state_repr = "\\texttt{True}" if state_val else "\\texttt{False}"
            else:
                 state_repr = str(state_val)
            target_str = f"{var_name}={state_repr}"

        # Format condition part
        condition_str = ""
        if all_conditions:
            formatted_conditions = [self._format_latex_condition(cond) for cond in all_conditions]
            condition_str = " \\mid " + ", ".join(formatted_conditions)

        # Combine and return
        return f"P({target_str}{condition_str})"


    def __call__(self, query_input: Union[Variable, VariableCombination, TargetStateQuery, QueryExpression]) -> Union[List[Dict[str, Any]], float]:
        """
        Calculates probability based on the input expression.
        (Implementation remains the same as previous version - code omitted for brevity)
        """
        # --- This method's internal logic is unchanged from the previous version ---
        # ... Parses input ...
        # ... Separates evidence, do_equality_dict, do_range_conditions, range_conditions ...
        # ... Determines query_vars ...
        # ... Handles deterministic case ...
        # ... Performs CausalInference or VariableElimination query ...
        # ... Applies range condition filtering ...
        # ... Marginalizes ...
        # ... Normalizes ...
        # ... Extracts float or Formats list output ...

        # --- Re-paste full __call__ implementation for completeness ---
        target_obj: Optional[Union[Variable, VariableCombination, TargetStateQuery]] = None
        all_conditions: List[Union[Condition, DoCondition, DoRangeCondition]] = []
        dag_instance: Optional[DAG] = None
        is_specific_state_query = False

        # 1. Parse input
        if isinstance(query_input, (Variable, VariableCombination)): target_obj = query_input; dag_instance = query_input._dag
        elif isinstance(query_input, TargetStateQuery): target_obj = query_input; dag_instance = query_input._dag; is_specific_state_query = True
        elif isinstance(query_input, QueryExpression):
            target_obj = query_input.target; dag_instance = query_input._dag; all_conditions = query_input.conditions
            if isinstance(target_obj, TargetStateQuery): is_specific_state_query = True
        else: 
            raise TypeError("Input to P() must be a Variable, VariableCombination, TargetStateQuery, or QueryExpression.")

        # Separate conditions
        equality_evidence = {}; do_equality_dict = {}; do_range_conditions = []; range_conditions = []
        has_intervention = False
        for cond in all_conditions:
            if isinstance(cond, DoCondition):
                has_intervention = True; var, val = cond.variable_name, cond.value
                if var in do_equality_dict and do_equality_dict[var] != val: raise ValueError(f"Conflicting do(==) interventions for '{var}'.")
                if var in equality_evidence: raise ValueError(f"Cannot have '{var}' in both evidence and do(==) intervention.")
                do_equality_dict[var] = val
            elif isinstance(cond, DoRangeCondition):
                 has_intervention = True; do_range_conditions.append(cond)
            elif isinstance(cond, Condition):
                if cond.operator == '==':
                    var, val = cond.variable_name, cond.value
                    if var in equality_evidence and equality_evidence[var] != val: raise ValueError(f"Conflicting equality evidence for '{var}'.")
                    if var in do_equality_dict: raise ValueError(f"Cannot have '{var}' in both evidence and do(==) intervention.")
                    equality_evidence[var] = val
                elif cond.operator in ['!=', '>', '>=', '<', '<=']: range_conditions.append(cond)
                else: raise NotImplementedError(f"Condition operator '{cond.operator}' not supported.")

        # 2. Determine query variables
        if target_obj is None: raise ValueError("Query target is missing.")
        underlying_target_names: List[str]
        if isinstance(target_obj, Variable): underlying_target_names = [target_obj.name]
        elif isinstance(target_obj, VariableCombination): underlying_target_names = [v.name for v in target_obj.variables]
        elif isinstance(target_obj, TargetStateQuery): underlying_target_names = [target_obj.variable.name]
        else: raise TypeError("Internal Error: Invalid target_obj type")
        all_range_vars = list(set([rc.variable_name for rc in range_conditions] + [drc.variable_name for drc in do_range_conditions]))
        query_vars = list(dict.fromkeys(underlying_target_names + all_range_vars))
        query_vars = [v for v in query_vars if v not in do_equality_dict]

        # Handle deterministic case
        if not query_vars:
             print("Warning: Target variable(s) are fixed by do(==) intervention. Result is deterministic.")
             if is_specific_state_query and isinstance(target_obj, TargetStateQuery):
                 target_var, target_state = target_obj.variable.name, target_obj.state
                 return 1.0 if target_var in do_equality_dict and do_equality_dict[target_var] == target_state else 0.0
             return [] if not is_specific_state_query else 0.0

        # 3. Perform main query
        try:
            initial_joint_factor: DiscreteFactor
            if has_intervention:
                causal_inference = CausalInference(dag_instance.model)
                initial_joint_factor = causal_inference.query(variables=query_vars, do=do_equality_dict, evidence=equality_evidence if equality_evidence else None)
            else:
                initial_joint_factor = dag_instance.inference.query(variables=query_vars, evidence=equality_evidence if equality_evidence else None, show_progress=False)

            # 4. Apply ALL range conditions via filtering
            all_range_conditions = range_conditions + do_range_conditions
            if all_range_conditions:
                # print(f"  Applying range conditions by filtering factor over {initial_joint_factor.variables}...") # Reduce noise
                filtered_factor = initial_joint_factor.copy(); factor_vars_ordered = filtered_factor.variables
                state_name_map = filtered_factor.state_names; state_combinations = [state_name_map[var] for var in factor_vars_ordered]
                flat_probabilities = filtered_factor.values.flatten(); new_probabilities = flat_probabilities.copy()
                prob_idx = 0
                for state_tuple in itertools.product(*state_combinations):
                    state_dict = dict(zip(factor_vars_ordered, state_tuple)); include_this_combination = True
                    for rc in all_range_conditions: # Check against ALL range conditions
                        var, op, val, current_state = rc.variable_name, rc.operator, rc.value, state_dict[rc.variable_name]
                        state_matches_condition = False
                        try: # Simplified check logic
                            state_numeric = None; value_is_numeric = isinstance(val, (int, float))
                            try: state_numeric = float(current_state)
                            except ValueError: pass
                            if value_is_numeric and state_numeric is not None:
                                if op == '>': state_matches_condition = state_numeric > val
                                elif op == '>=': state_matches_condition = state_numeric >= val
                                elif op == '<': state_matches_condition = state_numeric < val
                                elif op == '<=': state_matches_condition = state_numeric <= val
                                elif op == '!=': state_matches_condition = not np.isclose(state_numeric, val)
                                elif op == '==': state_matches_condition = np.isclose(state_numeric, val)
                            elif isinstance(val, str) and state_numeric is None:
                                if op == '!=': state_matches_condition = (str(current_state) != str(val))
                                elif op == '==': state_matches_condition = (str(current_state) == str(val))
                        except Exception: pass
                        if not state_matches_condition: include_this_combination = False; break
                    if not include_this_combination: new_probabilities[prob_idx] = 0.0
                    prob_idx += 1
                filtered_factor.values = new_probabilities.reshape(filtered_factor.cardinality)

                # 5. Marginalize
                vars_to_marginalize = [rv for rv in all_range_vars if rv not in underlying_target_names]
                if vars_to_marginalize:
                    # print(f"  Marginalizing out range variables: {vars_to_marginalize}") # Reduce noise
                    final_factor = filtered_factor.marginalize(vars_to_marginalize, inplace=False)
                else: final_factor = filtered_factor

                # 6. Normalize
                # print("  Normalizing final factor...") # Reduce noise
                if np.sum(final_factor.values) > 1e-10: final_factor.normalize(inplace=True)
                else: print("Warning: Probability sum is zero after applying range conditions."); final_factor.values[:] = 0.0
            else:
                final_factor = initial_joint_factor

            # 7. Format output OR Extract specific probability
            if is_specific_state_query:
                 target_state_query = target_obj
                 target_var_name = target_state_query.variable.name
                 desired_state = target_state_query.state
                 if target_var_name not in final_factor.variables:
                      print(f"Warning: Target variable '{target_var_name}' not found in final factor.")
                      return 0.0
                 probability = 0.0; final_factor_vars = final_factor.variables
                 final_state_combinations = [final_factor.state_names[var] for var in final_factor_vars]
                 final_probabilities = final_factor.values.flatten(); prob_idx = 0
                 target_var_index = final_factor_vars.index(target_var_name)
                 for state_tuple in itertools.product(*final_state_combinations):
                      current_state_for_target = state_tuple[target_var_index]
                      actual_desired_state = desired_state
                      if target_var_name in dag_instance._bool_cols:
                           if isinstance(desired_state, bool): current_state_for_target = (current_state_for_target.lower() == 'true')
                      if current_state_for_target == actual_desired_state: probability += final_probabilities[prob_idx]
                      prob_idx += 1
                 return float(probability)
            else: # Format full distribution
                output_list = []; final_factor_vars = final_factor.variables
                final_state_combinations = [final_factor.state_names[var] for var in final_factor_vars]
                final_probabilities = final_factor.values.flatten(); prob_idx = 0
                for state_tuple in itertools.product(*final_state_combinations):
                    row_dict = {}
                    for i, var_name in enumerate(final_factor_vars):
                        state_value = state_tuple[i]
                        if var_name in dag_instance._bool_cols:
                            if state_value.lower() == 'true': state_value = True
                            elif state_value.lower() == 'false': state_value = False
                        row_dict[var_name] = state_value
                    row_dict['probability'] = final_probabilities[prob_idx]
                    output_list.append(row_dict); prob_idx += 1
                return output_list

        except Exception as e:
            print(f"Error during inference (DAG: {dag_instance}) for query {repr(query_input)}: {e}")
            import traceback; traceback.print_exc(); raise


P = P_Calculator()


class DAG:
    """Represents a Directed Acyclic Graph (DAG) learned from data."""
    def __init__(self, nodes, edges, dataframe):
        if not isinstance(dataframe, pd.DataFrame): raise ValueError("dataframe must be a pandas DataFrame.")
        self.nodes = list(nodes)
        if not all(node in dataframe.columns for node in self.nodes):
            missing_nodes = [node for node in self.nodes if node not in dataframe.columns]
            raise ValueError(f"Nodes {missing_nodes} not found in DataFrame columns.")
        self.edges = edges
        self.dataframe = dataframe
        self._variables = {}

        self.model = DiscreteBayesianNetwork(ebunch=edges)
        self.model.add_nodes_from(self.nodes)

        df_copy = dataframe.copy()
        self._bool_cols = df_copy.select_dtypes(include=['bool']).columns.tolist()
        for col in self._bool_cols: 
            df_copy[col] = df_copy[col].astype(str)
        self._state_metadata = {}
        if 'age' in nodes: 
            self._state_metadata['age'] = {'type': 'numerical_bin'}
        self.model.fit(df_copy, estimator=MaximumLikelihoodEstimator)

        try:
            if not self.model.check_model(): 
                print("Warning: Model check reported issues.")
        except Exception as e: 
            print(f"Warning: Model check failed with an error: {e}.")

        self.inference = VariableElimination(self.model)

    def get_variables(self) -> List[Variable]:
        if len(self._variables) != len(self.nodes):
            temp_vars = {}
            for name in self.nodes:
                temp_vars.update({name: self._variables.get(name, Variable(name, self))})
            self._variables = temp_vars
        return [self._variables[name] for name in self.nodes]

    def _prepare_evidence(self, evidence_dict):
        prepared_evidence = {}
        for var, value in evidence_dict.items(): 
            prepared_evidence[var] = str(value) if var in self._bool_cols else value
        return prepared_evidence

    def P(self, target_variable, evidence=None) -> DiscreteFactor:
        target_name = target_variable.name if isinstance(target_variable, Variable) else target_variable
        if target_name not in self.nodes: 
            raise ValueError(f"Unknown target variable: {target_name}")
        prepared_evidence = self._prepare_evidence(evidence) if evidence else None
        try: 
            return self.inference.query(variables=[target_name], evidence=prepared_evidence, show_progress=False)
        except Exception as e: 
            print(f"Error during inference via DAG.P for P({target_name} | {prepared_evidence}): {e}")
            raise

