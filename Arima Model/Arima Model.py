#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


#  ## Data Gathering
# 

# In[6]:


from arima_utils.data_gathering import get_data
df = get_data('F')


# ## stationarity tests
# 

# In[7]:


import warnings
warnings.filterwarnings("ignore")
from arima_utils.stationarity_tests import rolling_stats
rolling_stats(df)
from arima_utils.stationarity_tests import dickey_fuller
dickey_fuller(df)
from arima_utils.stationarity_tests import log_diff
log_diff(df)
from arima_utils.stationarity_tests import auto_correlation
auto_correlation(df)


# ## searching for best ARIMA order
# 

# In[8]:


warnings.filterwarnings("ignore")
# evaluate parameters
p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)
from arima_utils.best_model import evaluate_models
best_model = evaluate_models(df, p_values, d_values, q_values)


# ## using best arima found in making walk-forward predictions

# In[9]:


from arima_utils.predictions import one_step_prediction
predictions_series, test = one_step_prediction(df, best_model)


# ## accuracy metrics
# 

# In[10]:


from arima_utils.predictions import accuracy_metrics
accuracy_metrics(predictions_series, test)


# In[ ]:




