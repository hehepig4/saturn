"""
Voting Data Structures for Federated Primitive TBox

Dynamic Pydantic model creation for binary voting (0/1).
"""

from typing import List, Dict, Type
from pydantic import BaseModel, Field, create_model


def create_voting_model(class_names: List[str]) -> Type[BaseModel]:
    """Dynamically create a Pydantic model for binary voting.
    
    Each class gets a score field with value 0 or 1:
    - 0: Not useful for this cluster's data
    - 1: Useful for this cluster's data
    
    Args:
        class_names: List of class names to vote on
        
    Returns:
        Dynamically created Pydantic model class
        
    Example:
        >>> VotingModel = create_voting_model(["YearColumn", "NameColumn"])
        >>> vote = VotingModel(score_YearColumn=1, score_NameColumn=0)
        >>> vote.model_dump()
        {'score_YearColumn': 1, 'score_NameColumn': 0}
    """
    fields = {}
    for class_name in class_names:
        # Sanitize class name for field name (remove special chars)
        field_name = f"score_{_sanitize_name(class_name)}"
        fields[field_name] = (
            int,
            Field(
                ge=0,
                le=1,
                description=f"Usefulness score for class '{class_name}' (0=not useful, 1=useful)"
            )
        )
    
    return create_model('ClassVoting', **fields)


def _sanitize_name(name: str) -> str:
    """Sanitize class name for use as Python identifier.
    
    Replaces special characters with underscores.
    """
    return name.replace("-", "_").replace(" ", "_").replace(":", "_")


def parse_voting_result(
    voting_model_instance: BaseModel,
    class_names: List[str]
) -> Dict[str, int]:
    """Parse voting model instance to class_name -> score dict.
    
    Args:
        voting_model_instance: Instance of dynamically created voting model
        class_names: Original class names (before sanitization)
        
    Returns:
        Dict mapping original class_name to score (0 or 1)
    """
    result = {}
    model_dict = voting_model_instance.model_dump()
    
    for class_name in class_names:
        field_name = f"score_{_sanitize_name(class_name)}"
        if field_name in model_dict:
            result[class_name] = model_dict[field_name]
        else:
            # Default to 0 if not found
            result[class_name] = 0
    
    return result


def aggregate_votes(
    all_votes: Dict[str, Dict[str, int]],
    class_names: List[str]
) -> Dict[str, float]:
    """Aggregate votes from all Local Agents.
    
    Computes weighted average (currently uniform weights).
    
    Args:
        all_votes: {group_id: {class_name: score}}
        class_names: List of all class names
        
    Returns:
        {class_name: aggregated_score} where score is in [0, 1]
    """
    if not all_votes:
        return {name: 0.0 for name in class_names}
    
    n_groups = len(all_votes)
    aggregated = {}
    
    for class_name in class_names:
        total_score = sum(
            votes.get(class_name, 0)
            for votes in all_votes.values()
        )
        aggregated[class_name] = total_score / n_groups
    
    return aggregated
