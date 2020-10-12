#installing packages
if (!require("doBy")) install.packages("doBy")
if (!require("lme4")) install.packages("lme4")

#calling packages
library("doBy")
library(lme4)

setwd("")
input <- read.csv("input_lucas.csv", stringsAsFactors = T)
input$rep <- as.factor(input$rep)
input$genotype <- as.factor(input$genotype)

sum <- summaryBy(phenotype ~ genotype, data = input, FUN = c(mean, sd, length)) 
hist(sum$phenotype.mean, main = "The distribution of HM content", xlab = "HM content")

write.csv(sum, file = "HM_summary.csv")

#Quick estimation and data checking by Anova
fit <- lm(phenotype ~ genotype, data = input)
par(mfcol = c(2,2))
plot(fit) 
aov.s <- anova(fit)
nrep <- 3
varg <- (aov.s$`Mean Sq`[1] - aov.s$`Mean Sq`[2])/nrep 
vare <- aov.s$`Mean Sq`[2]
herit.s <- varg/(varg+vare)
herit.s 

#linear model H2 calculation 

fitlmer_rand <- lmer(phenotype ~ (1|genotype), data = input, REML = TRUE) # + (1|row) + (1|column) if you needed 
sum <- summary(fitlmer_rand)
out <- as.data.frame(VarCorr(fitlmer_rand))
i <- 1
if (i == 1){
  varcom <- out$vcov
  header <- out$grp
  varcom <- rbind(header, varcom)
  
} else {
  #varcom <- rbind(varcom, out.c$vcov, out.t$vcov)
  varcom <- rbind(varcom, out$vcov)
}

varcom
write.csv(varcom, file = "H2.csv")
