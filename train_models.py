import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, will skip this model")

# Try to import TabPFN
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("⚠️  TabPFN not available, will skip this model")

print("\n" + "="*80)
print("🏥 CHILD NUTRITION RISK DETECTOR - MODEL TRAINING SYSTEM")
print("="*80)
print("📍 Location: D:\\Safwan\\projects\\CNRD")
print("="*80)

# Create directories
os.makedirs(os.path.join("models", "saved_models"), exist_ok=True)
print("✅ Created directories: models/saved_models/")

# Generate synthetic training data
def generate_training_data(n_samples=2000):
    """
    Generate synthetic nutrition risk data for training
    Features: 20 parameters
    Target: 0 (Healthy), 1 (At Risk)
    """
    np.random.seed(42)
    
    print(f"\n📊 Generating {n_samples} synthetic training samples...")
    
    data = []
    
    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")
        
        # Age (0-18 years)
        age = np.random.uniform(0.5, 18)
        
        # Gender (0: Male, 1: Female)
        gender = np.random.randint(0, 2)
        
        # Weight (kg) - age-dependent with variation
        base_weight = age * 3 + 3
        weight = max(2.5, base_weight + np.random.normal(0, 3))
        
        # Height (cm) - age-dependent with variation
        base_height = age * 6 + 50
        height = max(50, base_height + np.random.normal(0, 5))
        
        # BMI
        bmi = weight / ((height/100) ** 2)
        
        # Mother's Education (0-3: None, Primary, Secondary, Higher)
        mother_edu = np.random.choice([0, 1, 2, 3], p=[0.15, 0.35, 0.35, 0.15])
        
        # Household Income (1000-50000 INR)
        income = np.random.lognormal(9, 0.8)
        income = min(max(income, 1000), 50000)
        
        # Meals per Day (1-5)
        meals = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.40, 0.30, 0.10])
        
        # Vaccination Status (0: Incomplete, 1: Complete)
        vaccination = np.random.choice([0, 1], p=[0.20, 0.80])
        
        # Access to Clean Water (0: No, 1: Yes)
        clean_water = np.random.choice([0, 1], p=[0.25, 0.75])
        
        # Region (0: Urban, 1: Rural, 2: Suburban)
        region = np.random.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
        
        # Birth Weight (1.5-4.5 kg)
        birth_weight = np.random.normal(3.0, 0.5)
        birth_weight = min(max(birth_weight, 1.5), 4.5)
        
        # Family Size (2-10)
        family_size = np.random.choice(range(2, 11), p=[0.10, 0.15, 0.25, 0.20, 0.15, 0.08, 0.04, 0.02, 0.01])
        
        # Father's Education (0-3)
        father_edu = np.random.choice([0, 1, 2, 3], p=[0.12, 0.33, 0.38, 0.17])
        
        # Food Habits (0: Veg, 1: Non-veg, 2: Mixed)
        food_habits = np.random.choice([0, 1, 2], p=[0.30, 0.25, 0.45])
        
        # Inherited Diseases (0-3)
        diseases = np.random.choice([0, 1, 2, 3], p=[0.70, 0.15, 0.10, 0.05])
        
        # Appetite Level (1-5)
        appetite = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.25, 0.40, 0.20])
        
        # Place of Birth (0-3: Home, Hospital, Clinic, Other)
        place_birth = np.random.choice([0, 1, 2, 3], p=[0.20, 0.60, 0.15, 0.05])
        
        # Sanitation Access (0: No, 1: Yes)
        sanitation = np.random.choice([0, 1], p=[0.30, 0.70])
        
        # Breastfeeding Duration (0-36 months)
        breastfeeding = np.random.gamma(2, 4)
        breastfeeding = min(breastfeeding, 36)
        
        # ==================== RISK SCORING LOGIC ====================
        risk_score = 0
        
        # 1. BMI-based risk (age-adjusted)
        if age > 2:
            if age <= 5:
                if bmi < 14:
                    risk_score += 2
            elif age <= 10:
                if bmi < 14.5:
                    risk_score += 2
            else:
                if bmi < 15.5:
                    risk_score += 2
        
        # 2. Low birth weight
        if birth_weight < 2.5:
            risk_score += 2
        
        # 3. Insufficient meals
        if meals < 3:
            risk_score += 1.5
        
        # 4. Low household income
        if income < 5000:
            risk_score += 1.5
        elif income < 10000:
            risk_score += 0.5
        
        # 5. No clean water
        if clean_water == 0:
            risk_score += 1
        
        # 6. No sanitation
        if sanitation == 0:
            risk_score += 1
        
        # 7. Low parental education
        if mother_edu == 0:
            risk_score += 1
        if father_edu == 0:
            risk_score += 0.5
        
        # 8. Incomplete vaccination
        if vaccination == 0:
            risk_score += 1.5
        
        # 9. Poor appetite
        if appetite <= 2:
            risk_score += 1
        
        # 10. Inadequate breastfeeding
        if breastfeeding < 6:
            risk_score += 1.5
        
        # 11. Large family with low income
        if family_size > 6 and income < 15000:
            risk_score += 1
        
        # 12. Inherited diseases present
        if diseases > 0:
            risk_score += 0.5 * diseases
        
        # 13. Rural area with poor facilities
        if region == 1 and (clean_water == 0 or sanitation == 0):
            risk_score += 0.5
        
        # 14. Home birth (may indicate limited healthcare access)
        if place_birth == 0:
            risk_score += 0.5
        
        # Determine target (At Risk if score >= 5)
        target = 1 if risk_score >= 5 else 0
        
        data.append([
            age, gender, weight, height, bmi, mother_edu, income, meals,
            vaccination, clean_water, region, birth_weight, family_size,
            father_edu, food_habits, diseases, appetite, place_birth,
            sanitation, breastfeeding, target
        ])
    
    columns = [
        'age', 'gender', 'weight', 'height', 'bmi', 'mother_edu', 'income',
        'meals', 'vaccination', 'clean_water', 'region', 'birth_weight',
        'family_size', 'father_edu', 'food_habits', 'diseases', 'appetite',
        'place_birth', 'sanitation', 'breastfeeding', 'target'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate data
df = generate_training_data(2000)

# Save dataset
df.to_csv('training_data.csv', index=False)
print(f"\n✅ Generated {len(df)} samples and saved to 'training_data.csv'")

# Display class distribution
healthy_count = (df['target']==0).sum()
at_risk_count = (df['target']==1).sum()
print(f"\n📈 Class Distribution:")
print(f"  🟢 Healthy: {healthy_count} samples ({healthy_count/len(df)*100:.1f}%)")
print(f"  🔴 At Risk: {at_risk_count} samples ({at_risk_count/len(df)*100:.1f}%)")

# Prepare data
X = df.drop('target', axis=1).values
y = df['target'].values

print(f"\n📦 Dataset Shape:")
print(f"  Features (X): {X.shape}")
print(f"  Target (y): {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️  Train/Test Split:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# Scale features
print(f"\n⚙️  Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_path = os.path.join("models", "saved_models", "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# Define models
print(f"\n🤖 Initializing Machine Learning Models...")

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVC": SVC(probability=True, random_state=42, kernel='rbf'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=8),
    "NaiveBayes": GaussianNB(),
    "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=50),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(random_state=42, n_estimators=100)
    print("  ✅ XGBoost included")

# Add TabPFN if available
if TABPFN_AVAILABLE:
    models["TabPFN"] = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
    print("  ✅ TabPFN included")

print(f"  📊 Total models to train: {len(models)}")

# Train and evaluate models
print(f"\n" + "="*80)
print("🏋️  TRAINING MODELS...")
print("="*80 + "\n")

results = []

for idx, (name, model) in enumerate(models.items(), 1):
    print(f"[{idx}/{len(models)}] Training {name}...", end=" ")
    try:
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Save model
        model_path = os.path.join("models", "saved_models", f"{name}.pkl")
        joblib.dump(model, model_path)
        
        results.append({
            'Model': name,
            'Train Accuracy': f"{train_score*100:.2f}%",
            'Test Accuracy': f"{test_score*100:.2f}%",
            'Status': '✅'
        })
        
        print(f"✅ Train: {train_score*100:.1f}% | Test: {test_score*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        results.append({
            'Model': name,
            'Train Accuracy': 'N/A',
            'Test Accuracy': 'N/A',
            'Status': '❌'
        })

# Display results
print("\n" + "="*80)
print("📈 TRAINING RESULTS SUMMARY")
print("="*80 + "\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

successful_models = sum(1 for r in results if r['Status'] == '✅')
print(f"\n✅ Successfully trained: {successful_models}/{len(models)} models")

print("\n" + "="*80)
print("🎉 MODEL TRAINING COMPLETED!")
print("="*80)
print(f"📁 Models saved in: models/saved_models/")
print(f"📊 Training data saved: training_data.csv")
print(f"🔧 Scaler saved: {scaler_path}")
print("\n💡 Next step: Run 'python app.py' to start the web application")
print("="*80 + "\n")