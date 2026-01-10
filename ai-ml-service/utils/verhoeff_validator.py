

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VerhoeffValidator:
    
    
    # Multiplication table (dihedral group D5)
    MULTIPLICATION_TABLE = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    # Permutation table
    PERMUTATION_TABLE = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]
    
    # Inverse table
    INVERSE_TABLE = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    
    @classmethod
    def validate(cls, number: str) -> bool:
        
        try:
            # Remove spaces and validate format
            number = number.replace(' ', '').replace('-', '')
            
            if not number.isdigit():
                logger.warning(f"Invalid Aadhaar format: contains non-digits")
                return False
            
            if len(number) != 12:
                logger.warning(f"Invalid Aadhaar length: {len(number)} (expected 12)")
                return False
            
            # Convert to list of integers
            digits = [int(d) for d in number]
            
            # Calculate checksum
            checksum = cls._calculate_checksum(digits)
            
            # Valid if checksum is 0
            is_valid = checksum == 0
            
            if is_valid:
                logger.debug(f"Aadhaar validation passed: {number[:4]}********")
            else:
                logger.debug(f"Aadhaar validation failed: checksum={checksum}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Verhoeff validation error: {e}")
            return False
    
    @classmethod
    def _calculate_checksum(cls, digits: list) -> int:
        
        checksum = 0
        
        # Process each digit from right to left
        for i, digit in enumerate(reversed(digits)):
            # Get permutation based on position
            permutation_index = i % 8
            permuted_digit = cls.PERMUTATION_TABLE[permutation_index][digit]
            
            # Apply multiplication table
            checksum = cls.MULTIPLICATION_TABLE[checksum][permuted_digit]
        
        return checksum
    
    @classmethod
    def generate_checksum_digit(cls, number: str) -> Optional[str]:
        
        try:
            # Validate input
            number = number.replace(' ', '').replace('-', '')
            
            if not number.isdigit() or len(number) != 11:
                logger.error(f"Invalid input for checksum generation: {number}")
                return None
            
            # Try each digit 0-9 as checksum
            for check_digit in range(10):
                test_number = number + str(check_digit)
                if cls.validate(test_number):
                    logger.debug(f"Generated valid Aadhaar: {test_number[:4]}********")
                    return test_number
            
            logger.warning("Could not generate valid checksum")
            return None
            
        except Exception as e:
            logger.error(f"Checksum generation error: {e}")
            return None
    
    @classmethod
    def format_aadhaar(cls, number: str) -> str:
        
        # Remove existing spaces/separators
        clean = number.replace(' ', '').replace('-', '')
        
        if len(clean) != 12:
            return number  # Return as-is if invalid length
        
        # Format: XXXX XXXX XXXX
        return f"{clean[0:4]} {clean[4:8]} {clean[8:12]}"
    
    @classmethod
    def mask_aadhaar(cls, number: str, show_last: int = 4) -> str:
        
        clean = number.replace(' ', '').replace('-', '')
        
        if len(clean) != 12:
            return "XXXX XXXX XXXX"  # Default mask if invalid
        
        # Show only last N digits
        masked = 'X' * (12 - show_last) + clean[-show_last:]
        
        # Format with spaces
        return f"{masked[0:4]} {masked[4:8]} {masked[8:12]}"


# Convenience function for direct validation
def validate_aadhaar(number: str) -> bool:
    
    return VerhoeffValidator.validate(number)


# Convenience function for formatting
def format_aadhaar(number: str) -> str:
    
    return VerhoeffValidator.format_aadhaar(number)


# Convenience function for masking
def mask_aadhaar(number: str, show_last: int = 4) -> str:
    
    return VerhoeffValidator.mask_aadhaar(number, show_last)


if __name__ == "__main__":
    # Test cases
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Verhoeff Validator Test ===\n")
    
    # Test valid Aadhaar numbers (these are examples, not real)
    test_cases = [
        ("234123412346", "Known valid"),
        ("123456789012", "Should be invalid"),
        ("2345 6789 0129", "With spaces"),
        ("919807654321", "Another test"),
    ]
    
    for number, description in test_cases:
        is_valid = validate_aadhaar(number)
        formatted = format_aadhaar(number)
        masked = VerhoeffValidator.mask_aadhaar(number)
        
        print(f"{description}")
        print(f"  Number: {formatted}")
        print(f"  Valid: {is_valid}")
        print(f"  Masked: {masked}")
        print()
