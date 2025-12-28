from typing import Any, Dict, List, Optional

class ChromaFilterBuilder:
    """
    Utility to build ChromaDB metadata filter dictionaries.
    Supports operators like $eq, $ne, $gt, $gte, $lt, $lte.
    """
    
    @staticmethod
    def build_filter(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Translates a dictionary of filter criteria into a ChromaDB-compatible query.
        
        Args:
            filters: Input dictionary (e.g., {"rating": {"$gte": 4.0}})
            
        Returns:
            A sanitized ChromaDB filter dictionary or None if no valid filters exist.
        """
        if not filters:
            return None
            
        filter_list = []
        for key, value in filters.items():
            if value is None:
                continue
                
            if isinstance(value, dict):
                # Handle operator-based filters: {"rating": {"$gte": 4}}
                # Ensure the operator value itself isn't None
                if any(v is None for v in value.values()):
                    continue
                filter_list.append({key: value})
            else:
                # Handle equality filters: {"restaurant": "Beyond Flavours"}
                filter_list.append({key: {"$eq": value}})
        
        if len(filter_list) == 0:
            return None
            
        if len(filter_list) == 1:
            return filter_list[0]
        
        return {"$and": filter_list}

if __name__ == "__main__":
    # Test cases
    builder = ChromaFilterBuilder()
    print(builder.build_filter({"restaurant": "Beyond Flavours"}))
    print(builder.build_filter({"restaurant": "Beyond Flavours", "rating": {"$gte": 4}}))
