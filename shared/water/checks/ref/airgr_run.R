# Reference GR4J runs with the real airGR package (CRAN 1.7.9, compiled in watersim/.refenv), example basin L0123001.
.libPaths(c(Sys.getenv("WATERSIM_RLIB", "../../watersim/.refenv/rlib"), .libPaths()))  # isolated reference env
library(airGR)
data(L0123001)
cat("airGR", as.character(packageVersion("airGR")), "\n")
InputsModel <- CreateInputsModel(FUN_MOD = RunModel_GR4J, DatesR = BasinObs$DatesR, Precip = BasinObs$P, PotEvap = BasinObs$E)
d <- format(BasinObs$DatesR, format = "%Y-%m-%d")
Param <- c(X1 = 257.238, X2 = 1.012, X3 = 88.235, X4 = 2.208)   # the parameter set of the airGR help example
# (1) the documented help example: run 1990-1999 with airGR's default warm-up (the preceding year) and initial stores
Ind_Run <- seq(which(d == "1990-01-01"), which(d == "1999-12-31"))
RunOptions <- CreateRunOptions(FUN_MOD = RunModel_GR4J, InputsModel = InputsModel, IndPeriod_Run = Ind_Run)
Out1 <- RunModel_GR4J(InputsModel = InputsModel, RunOptions = RunOptions, Param = Param)
InputsCrit <- CreateInputsCrit(FUN_CRIT = ErrorCrit_NSE, InputsModel = InputsModel, RunOptions = RunOptions, Obs = BasinObs$Qmm[Ind_Run])
c1 <- ErrorCrit_NSE(InputsCrit = InputsCrit, OutputsModel = Out1)
cat("help example NSE", c1$CritValue, "\n")
# (2) explicit initial states, no warm-up, 1989-01-01..1999-12-31 (identical start for an exact comparison)
Ind_All <- seq(which(d == "1989-01-01"), which(d == "1999-12-31"))
Ini <- CreateIniStates(FUN_MOD = RunModel_GR4J, InputsModel = InputsModel, ProdStore = 0.3 * Param[["X1"]],
                       RoutStore = 0.5 * Param[["X3"]], UH1 = rep(0, 20), UH2 = rep(0, 40))
RO2 <- CreateRunOptions(FUN_MOD = RunModel_GR4J, InputsModel = InputsModel, IndPeriod_Run = Ind_All,
                        IndPeriod_WarmUp = 0L, IniStates = Ini)
Out2 <- RunModel_GR4J(InputsModel = InputsModel, RunOptions = RO2, Param = Param)
write.csv(data.frame(date = d[Ind_All], P = BasinObs$P[Ind_All], E = BasinObs$E[Ind_All], Qobs = BasinObs$Qmm[Ind_All],
                     Qsim = Out2$Qsim, Prod = Out2$Prod, Rout = Out2$Rout, AE = Out2$AE, Exch = Out2$Exch),
          "airgr_gr4j_explicit.csv", row.names = FALSE)
write.csv(data.frame(date = d[Ind_Run], Qsim = Out1$Qsim), "airgr_gr4j_help_example.csv", row.names = FALSE)
# full input series (1984-) for our own default-warm-up run
write.csv(data.frame(date = d, P = BasinObs$P, E = BasinObs$E, Qobs = BasinObs$Qmm), "airgr_L0123001_inputs.csv", row.names = FALSE)
cat(sprintf("help_nse=%.6f\n", c1$CritValue), file = "airgr_meta.txt")
cat(sprintf("area_km2=%s\n", BasinInfo$BasinArea), file = "airgr_meta.txt", append = TRUE)
