"""
Execution launcher for Credit Card Fraud Detection Web Application
"""

import sys
import os

if __name__ == "__main__":
    from app import app, get_predictor

    predictor = get_predictor()
    print("=" * 65)
    print("  Credit Card Fraud Detection System")
    print("  Powered by Ensemble Machine Learning")
    print("=" * 65)
    print(f"  Model Loaded   : {predictor.is_loaded}")
    print(f"  Algorithm      : {predictor.model_name}")
    print(f"  Risk Threshold : {predictor.threshold}")
    print("  Local Server   : http://127.0.0.1:5000")
    print("=" * 65)
    print("  Press Ctrl+C to terminate the server.\n")

    app.run(host="127.0.0.1", port=5000, debug=False)
