"""Blood Analysis Module - Simplified Version"""

class BloodAnalyzer:
    def __init__(self):
        print("BloodAnalyzer initialized (simplified mode)")
    
    def analyze_blood_parameters(self, blood_values):
        """Simple blood analysis"""
        try:
            if not blood_values or all(v == 0 for v in blood_values):
                return 0.0
            
            # Simple risk calculation
            risk_score = 0.0
            
            # Hemoglobin check (index 0)
            if len(blood_values) > 0 and blood_values[0] > 0:
                if blood_values[0] < 10:
                    risk_score += 0.3
                elif blood_values[0] < 11:
                    risk_score += 0.15
            
            # Albumin check (index 9)
            if len(blood_values) > 9 and blood_values[9] > 0:
                if blood_values[9] < 3.0:
                    risk_score += 0.3
                elif blood_values[9] < 3.5:
                    risk_score += 0.15
            
            # Protein check (index 8)
            if len(blood_values) > 8 and blood_values[8] > 0:
                if blood_values[8] < 6.0:
                    risk_score += 0.2
            
            return min(risk_score, 1.0)
        except Exception as e:
            print(f"Blood analysis error: {e}")
            return 0.0
    
    def get_parameter_interpretation(self, blood_values):
        """Simple interpretation"""
        interpretations = {}
        
        param_names = ['Hemoglobin', 'RBC Count', 'WBC Count', 'Platelet Count',
                      'Hematocrit', 'MCV', 'MCH', 'MCHC', 'Protein', 'Albumin']
        
        for i, (name, value) in enumerate(zip(param_names, blood_values)):
            if value > 0:
                status = "Normal"
                if i == 0:  # Hemoglobin
                    if value < 10:
                        status = "Low"
                    elif value < 11:
                        status = "Below Normal"
                elif i == 9:  # Albumin
                    if value < 3.0:
                        status = "Low"
                    elif value < 3.5:
                        status = "Below Normal"
                
                interpretations[name] = {
                    'value': f"{value}",
                    'status': status,
                    'normal_range': 'N/A',
                    'clinical_significance': 'Basic screening only'
                }
        
        return interpretations