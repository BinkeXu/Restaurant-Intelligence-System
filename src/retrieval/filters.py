from typing import Any, Dict, List, Optional

class ChromaFilterBuilder:
    """
    Utility to build ChromaDB metadata filter dictionaries.
    Supports operators like $eq, $ne, $gt, $gte, $lt, $lte.
    """
    
    @staticmethod
    def build_filter(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Converts a simple dict of key-value pairs or key-operator pairs into ChromaDB format.
        Example input: {"restaurant": "Beyond Flavours", "rating": {"$gte": 4}}
        """
        if not filters:
            return None
            
        filter_list = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operator-based filters: {"rating": {"$gte": 4}}
                filter_list.append({key: value})
            else:
                # Handle equality filters: {"restaurant": "Beyond Flavours"}
                filter_list.append({key: {"$eq": value}})
        
        if len(filter_list) == 1:
            return filter_list[0]
        
        return {"$and": filter_list}

if __name__ == "__main__":
    # Test cases
    builder = ChromaFilterBuilder()
    print(builder.build_filter({"restaurant": "Beyond Flavours"}))
    print(builder.build_filter({"restaurant": "Beyond Flavours", "rating": {"$gte": 4}}))
