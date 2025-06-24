import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm



# Load dataset
statistic_sample_backward = torch.load("/content/generative-models/statistic_sample_backward.tar")
print(f'len(statistic_sample_backward): {len(statistic_sample_backward)}')
print(f'statistic_sample_backward: {statistic_sample_backward[:20]}')



in_model_layer = np.array([])
in_model_Unet_up_or_down_layer = np.array([])
in_model_position = np.array([])
in_model_replaced_module = np.array([])
current_timestep = np.array([])
grad_input_mean_abs = np.array([])
grad_output_mean_abs = np.array([])

for sample in statistic_sample_backward:
  in_model_layer = np.append(in_model_layer, sample["in_model_layer"])
  in_model_Unet_up_or_down_layer = np.append(in_model_Unet_up_or_down_layer, sample["in_model_Unet_up_or_down_layer"])
  in_model_position = np.append(in_model_position, sample["in_model_position"])
  in_model_replaced_module = np.append(in_model_replaced_module, sample["in_model_replaced_module"])
  current_timestep_new = sample["current_timestep"].mean().cpu().numpy()
  current_timestep = np.append(current_timestep, current_timestep_new)

  if sample["grad_input_mean_abs"][0] is not None:
    grad_input_mean_abs_new = sample["grad_input_mean_abs"][0].cpu().numpy()
    grad_input_mean_abs = np.append(grad_input_mean_abs, grad_input_mean_abs_new)
  else:
    grad_input_mean_abs = np.append(grad_input_mean_abs, None)

  grad_output_mean_abs_new = sample["grad_output_mean_abs"].cpu().numpy()
  grad_output_mean_abs = np.append(grad_output_mean_abs, grad_output_mean_abs_new)



in_model_position_dummy = pd.get_dummies(in_model_position, dtype=int).values
in_model_replaced_module_dummy = pd.get_dummies(in_model_replaced_module, dtype=int).values
print(in_model_position_dummy[:20])
print(in_model_replaced_module_dummy[:20])

X = np.column_stack([in_model_layer, in_model_Unet_up_or_down_layer, in_model_position_dummy[:, 0], current_timestep])
print(X[:20])

data = {
    'in_model_layer': in_model_layer, 
    'in_model_Unet_up_or_down_layer': in_model_Unet_up_or_down_layer, 
    'in_model_position': in_model_position, 
    'in_model_position_dummy0': in_model_position_dummy[:, 0],
    'in_model_position_dummy1': in_model_position_dummy[:, 1],
    'in_model_replaced_module': in_model_replaced_module, 
    'in_model_replaced_module_dummy0': in_model_replaced_module_dummy[:, 0], 
    'in_model_replaced_module_dummy1': in_model_replaced_module_dummy[:, 1], 
    'current_timestep': current_timestep, 
    'grad_input_mean_abs': grad_input_mean_abs, 
    'grad_output_mean_abs': grad_output_mean_abs, 
       }
data_df = pd.DataFrame(data)
data_df['in_model_layer'] = pd.to_numeric(data_df['in_model_layer'])
data_df['in_model_Unet_up_or_down_layer'] = pd.to_numeric(data_df['in_model_Unet_up_or_down_layer'])
data_df['in_model_position_dummy0'] = pd.to_numeric(data_df['in_model_position_dummy0'])
data_df['in_model_position_dummy1'] = pd.to_numeric(data_df['in_model_position_dummy1'])
data_df['in_model_replaced_module_dummy0'] = pd.to_numeric(data_df['in_model_replaced_module_dummy0'])
data_df['in_model_replaced_module_dummy1'] = pd.to_numeric(data_df['in_model_replaced_module_dummy1'])
data_df['current_timestep'] = pd.to_numeric(data_df['current_timestep'])
data_df['grad_input_mean_abs'] = pd.to_numeric(data_df['grad_input_mean_abs'])
data_df['grad_output_mean_abs'] = pd.to_numeric(data_df['grad_output_mean_abs'])
print(data_df[:20])



#clean None data
data_df_clean = data_df.dropna()



#heatmap, regplot
heatmap_df = data_df_clean[['in_model_layer', 'in_model_position', 'grad_input_mean_abs']]
pivot = heatmap_df.pivot_table(index='in_model_position', columns='in_model_layer', values='grad_input_mean_abs', aggfunc='mean').astype(float)
sns.heatmap(pivot)
plt.show()

heatmap_df = data_df_clean[['in_model_layer', 'in_model_Unet_up_or_down_layer', 'grad_input_mean_abs']]
pivot = heatmap_df.pivot_table(index='in_model_Unet_up_or_down_layer', columns='in_model_layer', values='grad_input_mean_abs', aggfunc='mean').astype(float)
sns.heatmap(pivot)
plt.show()

sns.regplot(data=data_df_clean, x='current_timestep', y='grad_input_mean_abs')
plt.show()



heatmap_df = data_df[['in_model_layer', 'in_model_position', 'grad_output_mean_abs']]
pivot = heatmap_df.pivot_table(index='in_model_position', columns='in_model_layer', values='grad_output_mean_abs', aggfunc='mean').astype(float)
sns.heatmap(pivot)
plt.show()

heatmap_df = data_df[['in_model_layer', 'in_model_Unet_up_or_down_layer', 'grad_output_mean_abs']]
pivot = heatmap_df.pivot_table(index='in_model_Unet_up_or_down_layer', columns='in_model_layer', values='grad_output_mean_abs', aggfunc='mean').astype(float)
sns.heatmap(pivot)
plt.show()

sns.regplot(data=data_df, x='current_timestep', y='grad_output_mean_abs')
plt.show()



#ols
x = data_df_clean[['in_model_layer', 'in_model_Unet_up_or_down_layer', 'in_model_position_dummy0', 'current_timestep']]
y = data_df_clean['grad_input_mean_abs']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df_clean['in_model_layer']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df_clean['in_model_Unet_up_or_down_layer']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df_clean['in_model_position_dummy0']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df_clean['current_timestep']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())



x = data_df[['in_model_layer', 'in_model_Unet_up_or_down_layer', 'in_model_position_dummy0', 'current_timestep']]
y = data_df['grad_output_mean_abs']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df['in_model_layer']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df['in_model_Unet_up_or_down_layer']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df['in_model_position_dummy0']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

x = data_df['current_timestep']
x = sm.add_constant(x)
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())