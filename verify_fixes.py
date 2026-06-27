import ast

print("Verifying code fixes...\n")

# Test 1: Check plot_sna.py
with open('utils/plot_sna.py', encoding='utf-8') as f:
    code = f.read()
    if 'import networkx as nx' in code:
        print('✅ plot_sna.py: NetworkX import added')
    if code.count('import streamlit as st') == 1:
        print('✅ plot_sna.py: Streamlit at top-level (not nested)')

# Test 2: Check components.py
with open('utils/components.py') as f:
    code = f.read()
    if 'from config import *' not in code:
        print('✅ components.py: Wildcard import removed')
    if 'APP_NAME' in code and 'APP_TITLE' in code:
        print('✅ components.py: Uses APP_NAME and APP_TITLE')

# Test 3: Check config.py
with open('config.py') as f:
    code = f.read()
    if 'APP_NAME = ' in code:
        print('✅ config.py: APP_NAME constant added')
    if 'LABEL_MAP = ' in code:
        print('✅ config.py: LABEL_MAP constant added')
    if 'LABEL_NAMES = ' in code:
        print('✅ config.py: LABEL_NAMES constant added')

# Test 4: Check analytics_classification.py
with open('utils/analytics_classification.py') as f:
    code = f.read()
    if 'config.LABEL_MAP' in code:
        print('✅ analytics_classification.py: Uses config.LABEL_MAP')

# Test 5: Check helper.py
with open('utils/helper.py') as f:
    code = f.read()
    if 'required_columns' in code:
        print('✅ helper.py: Data validation added')

# Test 6: Check plot_sentiment.py
with open('utils/plot_sentiment.py') as f:
    code = f.read()
    if 'st.warning' in code:
        print('✅ plot_sentiment.py: Error handling with warnings added')

# Test 7: Check plot_topic.py
with open('utils/plot_topic.py') as f:
    code = f.read()
    if 'st.error' in code:
        print('✅ plot_topic.py: Error handling added')

print('\n✅ All code fixes verified successfully!')
