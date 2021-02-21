#installing packages
if (!require("doBy")) install.packages("doBy")
if (!require("lme4")) install.packages("lme4")
if (!require("lmtest")) install.packages("lmtest")
if (!require("heritability")) install.packages("heritability")

#calling packages
library("doBy")
library(lme4)
library(ggplot2)
library(lmtest)
library(heritability)


input <- read.csv("data/Noccaea_CQs.csv", stringsAsFactors = T)
input <- input[input$batch != "",] # Removes unprocessed (batch 3 + root rows)

input$rep <- as.factor(input$Biological.replicate)
input$genotype <- as.factor(input$Accession..)
phenotypes <- c("plant_n_pix", "plant_meanC", "petiole", "margin", "vein", "tissue", "rand_1", "rand_2", "rand_5")

metals <- c("metal_Zn", "metal_K", "metal_Ni", "metal_Ca")
# noise_levels <- c("10", "20", "50","75","90") 
# metal <- "metal_Z"
metric <- "CQ"


for (metal in metals){
# for (nlvl in noise_levels){
  result_labels <- paste(phenotypes, metal, sep="_")
  for (current_phenotype in phenotypes){
    if ((current_phenotype == "plant_n_pix") | (current_phenotype == "plant_meanC")){
      colname <- paste(metal, current_phenotype, sep = "_")
    }
    else{
      colname <- paste(metal, current_phenotype, metric, sep = "_")
    }
    input$phenotype <- input[,colname]
    # input$plant_meanC <- input[,paste(metal, "plant_meanC", sep = "_")]
    
    # Check assumptions
    fit <- lm(phenotype ~ genotype, data = input)
    par(mfcol = c(2,2))
    plot(fit)
    
    print((paste("below: ", colname)))
    test <- bptest(fit)
    if (test["p.value"] < .05){
      print(paste(colname, "not homoskedastic"))
      print(test["p.value"])
    }
    
    
    #mixed effects model H2 calculation
    par(mfrow=c(1,1)) 
    fitlmer_rand <- lmer(phenotype ~ (1|genotype), data = input, REML = TRUE) # + (1|row) + (1|column) if you needed 
    hist(input$phenotype, main=colname)
    print(plot(fitlmer_rand, main=colname))
    coef(fitlmer_rand, main=colname)
    qqnorm(resid(fitlmer_rand), main=colname)
    
    
    sum <- summary(fitlmer_rand)
    out <- as.data.frame(VarCorr(fitlmer_rand))
    H2_frac <- out$vcov[1] /  (out$vcov[1] + out$vcov[2])
    rep_sum <- repeatability(input$phenotype, input$genotype)
    print(paste("mixed H2", H2_frac))
    print(paste("Anova H2", rep_sum$repeatability))
    
    if (current_phenotype == phenotypes[1]){
      result <- data.frame(out$vcov[1], out$vcov[2])
      names(result) <- c("genotype", "residual")
    } else {
      result[nrow(result) + 1,] = list(out$vcov[1], out$vcov[2])
    }
    
    # mixed effects model H2 with plant meanC
    # fitlmer_rand_meanC <- lmer(phenotype ~ (1|genotype) + (1|plant_meanC), data = input, REML = TRUE)
    # out_meanC <- as.data.frame(VarCorr(fitlmer_rand_meanC))
  }
  
   
  result$total_var <- result$genotype + result$residual
  result$H2 <- result$genotype / result$total_var
  result$H2_percent <- result$H2 * 100
  row.names(result) <- result_labels
  if (metal == metals[1]){
    combined_results <- result
  }
  else{
    combined_results <- rbind(combined_results, result)
  }
  
  print(ggplot(data=result, aes(x=row.names(result), y=H2_percent, fill=row.names(result))) +
    geom_bar(stat="identity") +
    # scale_x_discrete(limits = phenotypes) +
    #scale_fill_brewer(palette="Dark2") +
    ylim(0,100) +
    xlab("phenotype attributes") +
    ylab("H2 (%)") +
    theme(axis.text.x = element_text(angle = 90)) +
    theme(legend.position = "none") +
    ggtitle(metal)) 
  # ggsave(paste0("data/output/plots/", metal, "_", metric, "_H2.png")) # Zimg_noise/H2.png
  
  # sorted_result <- with(result,  result[order(row.names(result)) , ])
}

write.csv(combined_results, "data/H2_CQ_table.csv")

# scatter plots for one phenotype
sampled_accessions <- sample(input$Accession.., 15)
input_plot <- input[input$Accession.. %in% sampled_accessions,]
input_plot$Accession.. <- as.factor(input_plot$Accession..)
ggplot(input_plot, aes(x=Accession.., y=metal_Ca_rand_2_CQ , color=Accession..)) + geom_point()
# 

# sum <- summaryBy(phenotype ~ genotype, data = input, FUN = c(mean, sd, length)) 
# hist(sum$phenotype.mean, main = paste("The distribution of ", current_phenotype), xlab = "HM content")

#write.csv(sum, file = "HM_summary.csv")

#Quick estimation and data checking by Anova
# fit <- lm(phenotype ~ genotype, data = input)
# par(mfcol = c(2,2))
# plot(fit, main = current_phenotype) 
# aov.s <- anova(fit)
# nrep <- 3
# varg <- (aov.s$`Mean Sq`[1] - aov.s$`Mean Sq`[2])/nrep 
# vare <- aov.s$`Mean Sq`[2]
# herit.s <- varg/(varg+vare)
# herit.s 

