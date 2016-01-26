data {
    int<lower=0> N; 
    int<lower=0> counts[N];
    real  exposures[N]; 
} 

parameters {
    real<lower=0> alpha; 
    real<lower=0> beta;
    real<lower=0> fluxes[N];
}

transformed parameters {
    real<lower=0> flux_cut;
    flux_cut <- 1./beta;
}

model {
    alpha ~ exponential(1.0);
    beta ~ gamma(0.1, 0.1);
    for (i in 1:N){
        fluxes[i] ~ gamma(alpha, beta);
        counts[i] ~ poisson(fluxes[i] * exposures[i]);
  }
}
