#installing packages
if (!require("doBy")) install.packages("doBy")
if (!require("lme4")) install.packages("lme4")
if (!require("lmtest")) install.packages("lmtest")

#calling packages
library("doBy")
library(lme4)
library(ggplot2)
library(lmtest)

input <- read.csv("data/inputCQ_size_meanC.csv", stringsAsFactors = T)

phenotypes <- c("plant_n_pix", "plant_meanC", "petiole", "margin", "vein", "tissue", "rand_5", "rand_10")
metals <- c("metal_Z", "metal_K", "metal_Ni", "metal_Ca")
metric <- "CQ"

for (metal in metals){
  result_labels <- paste(phenotypes, metal, sep="_")
  for (current_phenotype in phenotypes){
    if ((current_phenotype == "plant_n_pix") | (current_phenotype == "plant_meanC")){
      colname <- paste(metal, current_phenotype, sep = "_")
    }
    else{
      colname <- paste(metal, current_phenotype, metric, sep = "_")
    }
    input$phenotype <- input[,colname]

    # Check assumptions with linear model
    fit <- lm(phenotype ~ genotype, data = input)
    par(mfcol = c(2,2))
    plot(fit, main="check linear model")
    
    test <- bptest(fit)
    if (test["p.value"] < .05){
      print(paste(colname, "not homoskedastic"))
      print(test["p.value"])
    }
    
    #mixed effects model H2 calculation
    fitlmer_rand <- lmer(phenotype ~ (1|genotype), data = input, REML = TRUE) # + (1|row) + (1|column) if you needed 
    
    # Check assumptions mixed effects model
    par(mfrow=c(1,1))
    hist(input$phenotype, main=colname)
    print(plot(fitlmer_rand, main=paste("mixed model", colname))) # Plot residuals against fitted
    qqnorm(resid(fitlmer_rand), main=paste("mixed model",colname))
    
    
    #sum <- summary(fitlmer_rand)
    out <- as.data.frame(VarCorr(fitlmer_rand))
    if (current_phenotype == phenotypes[1]){
      result <- data.frame(out$vcov[1], out$vcov[2])
      names(result) <- c("genotype", "residual")
    } else {
      result[nrow(result) + 1,] = list(out$vcov[1], out$vcov[2])
    }
  }
  # Calculate H2
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
  
  ggplot(data=result, aes(x=row.names(result), y=H2_percent, fill=row.names(result))) +
    geom_bar(stat="identity") +
    scale_x_discrete(limits = phenotypes) +
    #scale_fill_brewer(palette="Dark2") +
    ylim(0,100) +
    xlab("phenotype attributes") +
    ylab("H2 (%)") +
    theme(axis.text.x = element_text(angle = 90)) +
    theme(legend.position = "none") +
    ggtitle(metal) 
  # ggsave(paste0("data/output/plots/", metal, "_", metric, "_H2.png")) # Zimg_noise/H2.png
}


