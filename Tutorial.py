import pandas as pd
import statsmodels.api as sm

dataset = pd.read_csv("advertising.csv")

media_spend = dataset[['TV', 'radio', 'newspaper']]
sales_data = dataset['sales']

media_spend = sm.add_constant(media_spend)

regression_model = sm.OLS(sales_data, media_spend).fit()

print(regression_model.summary())

residual_sum_squares = sum(regression_model.resid ** 2)
num_observations, num_predictors = media_spend.shape
residual_standard_error = (residual_sum_squares / (num_observations - num_predictors)) ** 0.5

f_stat = regression_model.fvalue
r_squared_value = regression_model.rsquared

print(f"\nResidual Standard Error (RSE): {residual_standard_error:.4f}")
print(f"R-squared: {r_squared_value:.4f}")
print(f"F-Statistic: {f_stat:.4f}")
