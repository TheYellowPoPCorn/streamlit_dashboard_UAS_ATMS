import traceback
import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print()

print("=" * 50)
print("Testing analytics_cluster import...")
print("=" * 50)

try:
    import utils.analytics_cluster as ac
    print("✅ Module imported")
    print("Module contents:", dir(ac))
    print()
    
    # Try to access run_kmeans
    if hasattr(ac, 'run_kmeans'):
        print("✅ run_kmeans found")
    else:
        print("❌ run_kmeans NOT found")
        
except Exception as e:
    print("❌ Error importing module:")
    traceback.print_exc()
    print()

print("=" * 50)
print("Testing direct import of run_kmeans...")
print("=" * 50)

try:
    from utils.analytics_cluster import run_kmeans
    print("✅ run_kmeans imported directly")
except Exception as e:
    print("❌ Error importing run_kmeans:")
    traceback.print_exc()
    print()

print("=" * 50)
print("Testing streamlit cache decorator...")
print("=" * 50)

try:
    import streamlit as st
    print("✅ Streamlit imported")
    
    # Test the decorator
    @st.cache_resource
    def test_func():
        return "test"
    
    print("✅ @st.cache_resource decorator works")
except Exception as e:
    print("❌ Error with streamlit:")
    traceback.print_exc()
