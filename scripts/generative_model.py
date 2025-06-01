import numpy as np
import pymc as pm

def fit_and_simulate(df, seed=45):
    np.random.seed(seed)

    # Encode churn as 0/1
    churn_flag = df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)

    # Dummy observed values
    df['logins'] = np.where(churn_flag == 1, np.random.poisson(3, size=len(df)), np.random.poisson(6, size=len(df)))
    df['support_contacts'] = np.where(churn_flag == 1, np.random.binomial(n=30, p=0.05, size=len(df)), np.random.binomial(n=30, p=0.02, size=len(df)))
    df['data_usage_gb'] = np.where(churn_flag == 1, np.random.gamma(2, 1.2, size=len(df)), np.random.gamma(2.5, 1.6, size=len(df)))

    with pm.Model() as model:
        # LOGINS
        lambda_churn = pm.Exponential("lambda_churn", lam=1.1)
        lambda_nochurn = pm.Exponential("lambda_nochurn", lam=1.05)
        lambda_logins = pm.math.switch(churn_flag, lambda_churn, lambda_nochurn)
        logins_obs = pm.Poisson("logins_obs", mu=lambda_logins, observed=df['logins'])

        # SUPPORT CONTACTS
        p_churn = pm.Beta("p_churn", alpha=2, beta=42)
        p_nochurn = pm.Beta("p_nochurn", alpha=2, beta=46)
        p_support = pm.math.switch(churn_flag, p_churn, p_nochurn)
        support_obs = pm.Binomial("support_obs", n=30, p=p_support, observed=df['support_contacts'])

        # DATA USAGE IN GB
        shape_churn = pm.Gamma("shape_churn", alpha=2, beta=1.3)
        scale_churn = pm.Gamma("scale_churn", alpha=2, beta=1.5)
        shape_nochurn = pm.Gamma("shape_nochurn", alpha=2, beta=1.25)
        scale_nochurn = pm.Gamma("scale_nochurn", alpha=2, beta=1.4)

        shape = pm.math.switch(churn_flag, shape_churn, shape_nochurn)
        scale = pm.math.switch(churn_flag, scale_churn, scale_nochurn)
        data_obs = pm.Gamma("data_obs", alpha=shape, beta=1/scale, observed=df['data_usage_gb'])

        trace = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=seed)

    # Draw posterior samples with noise
    lambda_churn_sample = np.random.choice(trace.posterior['lambda_churn'].values.flatten()) + np.random.normal(0, 0.05)
    lambda_nochurn_sample = np.random.choice(trace.posterior['lambda_nochurn'].values.flatten()) + np.random.normal(0, 0.05)

    p_churn_sample = np.clip(np.random.choice(trace.posterior['p_churn'].values.flatten()) + np.random.normal(0, 0.01), 0, 1)
    p_nochurn_sample = np.clip(np.random.choice(trace.posterior['p_nochurn'].values.flatten()) + np.random.normal(0, 0.01), 0, 1)

    shape_churn_sample = np.random.choice(trace.posterior['shape_churn'].values.flatten()) + np.random.normal(0, 0.1)
    scale_churn_sample = np.random.choice(trace.posterior['scale_churn'].values.flatten()) + np.random.normal(0, 0.1)

    shape_nochurn_sample = np.random.choice(trace.posterior['shape_nochurn'].values.flatten()) + np.random.normal(0, 0.1)
    scale_nochurn_sample = np.random.choice(trace.posterior['scale_nochurn'].values.flatten()) + np.random.normal(0, 0.1)


    # Simulate 30-day sequences based on churn
    logins = []
    support = []
    data_usage = []

    
    for is_churn in churn_flag:
        if is_churn: #If customer churns
            logins.append(np.random.poisson(lambda_churn_sample, size=30).tolist()) 
            support.append(np.random.binomial(n=1, p=p_churn_sample, size=30).tolist())
            data_usage.append(np.random.gamma(shape_churn_sample, scale_churn_sample, size=30).tolist())
        else: #If customer dont churn
            logins.append(np.random.poisson(lambda_nochurn_sample, size=30).tolist())
            support.append(np.random.binomial(n=1, p=p_nochurn_sample, size=30).tolist())
            data_usage.append(np.random.gamma(shape_nochurn_sample, scale_nochurn_sample, size=30).tolist())

    # Store sequences as list columns in dataframe
    df['logins_seq'] = logins
    df['support_seq'] = support
    df['data_seq'] = data_usage

    return df[['customerID', 'Churn', 'logins_seq', 'support_seq', 'data_seq']]
