/*
 * A model to estimate plant cover
 * from cover class data
 * using regularized incomplete beta function
 */

functions {
 /*
  * Return the log probability the cover class Y is observed
  * given the cut points CP and thebeta distribution parameters
  * a and b
  *
  * @param Y  Observed cover class (int)
  * @param CP Cut points (vector)
  * @param a  Parameteer of the beta distribution (real)
  * @param b  Parameteer of the beta distribution (real)
  *
  * @return Log probability that Y is observed
  */
  real coverclass_lpmf(int Y, vector CP, real a, real b) {
    int n_class;
    real gamma;

    n_class = num_elements(CP) + 1;
    if (Y <= 1) {  // 0 or 1
      gamma =  inc_beta(a, b, CP[1]);
    } else if(Y >= 2 && Y < n_class) {
      gamma = inc_beta(a, b, CP[Y])
              - inc_beta(a, b, CP[Y - 1]);
    } else {
      gamma = 1 - inc_beta(a, b, CP[n_class - 1]);
    }
    return bernoulli_lpmf(1 | gamma);
  }
}

data {
  int<lower = 1> N_q;                           // Number of quadrats
  int<lower = 1> N_y;                           // Number of years
  int<lower = 1> N_cls;                         // Number of classes
  int<lower = 1> N_obs;                         // Number of observed years
  int<lower = 1> Obs_y[N_obs];                  // Observed years
  vector<lower = 0, upper = 1>[N_cls - 1] CP;   // Cut point
  int<lower = 0, upper = N_cls> Y[N_obs, N_q];  // Cover class data
}

parameters {
  vector[N_y] theta;                        // Latent state
  matrix[N_obs, N_q] r;                     // Spatial random effect
  real<lower = 0, upper = 1> delta;         // Uncertainty in classification
  real<lower = 0> sigma[2];                 // Standard deviations of
                                            //   spatial random effect: sigma[1]
                                            //   and temporal variation: sigma[2]
}

model {
  // Spatial random effect
  for (i in 1:N_obs) {
    r[i, 1:(N_q - 1)] ~ normal(r[i, 2:N_q], sigma[1]);
    r[i, N_q] ~ normal(-sum(r[i, 1:(N_q - 1)]), sigma[1]);
  }

  // System model
  // Logit of coverage proportion
  theta[3:N_y] ~ normal(2 * theta[2:(N_y - 1)]
                          - theta[1:(N_y - 2)], sigma[2]);

  // Observation model
  for (i in 1:N_obs) {
    int y = Obs_y[i];
    for (q in 1:N_q) {
        real p = inv_logit(theta[y] + r[i, q]);
        real alpha = p / delta - p;
        real beta = (1 - p) * (1 - delta) / delta;
        
        Y[i, q] ~ coverclass(CP, alpha, beta);
    }
  }

  // Weakly informative priors
  theta[1:2] ~ normal(0, 5);
  sigma ~ normal(0, 2.5);
}

generated quantities {
  vector[N_y] phi = inv_logit(theta);
}
