# Reference HBV-type runs with TUWmodel (Parajka et al. 2007; CRAN 1.1-1), lumped Vils example (help page parameters).
.libPaths(c(Sys.getenv("WATERSIM_RLIB", "../../watersim/.refenv/rlib"), .libPaths()))  # isolated reference env
library(TUWmodel)
data(example_TUWmodel)
cat("TUWmodel", as.character(packageVersion("TUWmodel")), "\n")
P <- apply(P_Vils, 1, weighted.mean, w = areas_Vils)
Tt <- apply(T_Vils, 1, weighted.mean, w = areas_Vils)
E <- apply(PET_Vils, 1, weighted.mean, w = areas_Vils)
par <- c(1.02, 1.70, 2, 0, -0.336, 0.934, 121, 2.52, 0.473, 9.06, 142, 50.1, 2.38, 10, 25)
inc <- c(50, 0, 2.5, 2.5)
sim <- TUWmodel(prec = P, airt = Tt, ep = E, area = 1, param = par, incon = inc)
# routing switched off (bmax = 1) to expose the runoff generation without the triangular transformation
par2 <- par; par2[14] <- 1; par2[15] <- 0
sim2 <- TUWmodel(prec = P, airt = Tt, ep = E, area = 1, param = par2, incon = inc)
d <- names(P); if (is.null(d)) d <- rownames(P_Vils)
out <- data.frame(date = d, P = P, T = Tt, E = E, q = as.numeric(sim$q), swe = as.numeric(sim$swe),
                  moist = as.numeric(sim$moist), rain = as.numeric(sim$rain), snow = as.numeric(sim$snow),
                  melt = as.numeric(sim$melt), q0 = as.numeric(sim$q0), q1 = as.numeric(sim$q1), q2 = as.numeric(sim$q2),
                  eta = as.numeric(sim$eta), suz = as.numeric(sim$suz), slz = as.numeric(sim$slz),
                  q_norout = as.numeric(sim2$q))
write.csv(out, "tuw_vils.csv", row.names = FALSE)
# matched run for the WaterSim comparison: melt threshold Tm = (Tr + Ts)/2 = 1 C (HBV-light uses one threshold TT),
# no routing (bmax = 1) and with routing (bmax = 10, croute = 0: fixed triangular base)
par3 <- par; par3[5] <- 1.0; par3[14] <- 1; par3[15] <- 0
s3 <- TUWmodel(prec = P, airt = Tt, ep = E, area = 1, param = par3, incon = inc)
par4 <- par3; par4[14] <- 10
s4 <- TUWmodel(prec = P, airt = Tt, ep = E, area = 1, param = par4, incon = inc)
m <- data.frame(date = d, P = P, T = Tt, E = E, q = as.numeric(s3$q), q_route10 = as.numeric(s4$q), swe = as.numeric(s3$swe),
                moist = as.numeric(s3$moist), rain = as.numeric(s3$rain), snow = as.numeric(s3$snow),
                melt = as.numeric(s3$melt), eta = as.numeric(s3$eta), suz = as.numeric(s3$suz), slz = as.numeric(s3$slz),
                q0 = as.numeric(s3$q0), q1 = as.numeric(s3$q1), q2 = as.numeric(s3$q2))
write.csv(m, "tuw_vils_matched.csv", row.names = FALSE)
