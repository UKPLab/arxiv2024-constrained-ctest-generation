library(mgcv)
library(itsadug)
library(hash)
library(report)
library(xtable)
packageVersion("mgcv") # make sure that this is at least 1.8.32  

#############################
# Overall difficulties
#############################
# data loading
data = read.csv('r_data/data.csv')
data <- as.data.frame(unclass(data), stringsAsFactors = TRUE)
data$model = relevel(data$model, ref="mip")

print(data)

set.seed(0)
# simple difficulty model
difficulty_model <- bam(difficulty ~ model + tau + s(text, bs="re") + s(cefr, bs="re") + s(user_key, bs="re") + s(years) + model:tau, #
                        data=data,
                        method="fREML",
                        discrete=TRUE)
summary(difficulty_model)

saveRDS(difficulty_model, "r_models/difficulty.rds")
dependencies = c("mgcv")
file_obj <- file("r_models/difficulty.dep")
writeLines(dependencies, file_obj)
close(file_obj)

plot_smooth(difficulty_model, view="tau", plot_all="model", rm.ranef=TRUE, col=c('darkgreen', 'chocolate3', 'dodgerblue4','darkslategray4'),lwd = 2)
            
# Significance testing
set.seed(0)
difficulty_model_anova <- anova(difficulty_model)
print(difficulty_model_anova)

# Gam check, residuals
gam.check(difficulty_model)

# Individual reports
print(report(difficulty_model))

# Concurvity should be closer observed for values larger 0.8
print(concurvity(difficulty_model, full = FALSE))

# Pairwise comparisons for the model, check model:tau connections
wald_gam(difficulty_model, comp=list(model=levels(data$model)))

# Tex table:
print(xtable(difficulty_model_anova$pTerms.table))

# time model
set.seed(0)
time_model <- bam(time ~ model*tau + s(text, bs="re") + s(cefr, bs="re") + s(user_key, bs="re") + s(years),
                  data=data,
                  method="fREML",
                  discrete=TRUE)
summary(time_model)

saveRDS(time_model, "r_models/time.rds")
dependencies = c("mgcv")
file_obj <- file("r_models/time.dep")
writeLines(dependencies, file_obj)
close(file_obj)

# Significance testing
set.seed(0)
time_model_anova <- anova(time_model)
print(time_model_anova)

# Gam check, residuals
gam.check(time_model)

# Individual reports
print(report(time_model))

# Concurvity should be closer observed for values larger 0.8
print(concurvity(time_model, full = FALSE))

# Pairwise comparisons for the model, check model:tau connections
wald_gam(time_model, comp=list(model=levels(data$model)))

# Tex table:
print(xtable(time_model_anova$pTerms.table))

