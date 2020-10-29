#installing packages
if (!require("doBy")) install.packages("doBy")
if (!require("lme4")) install.packages("lme4")

#calling packages
library("doBy")
library(lme4)
library(ggplot2)

input <- read.csv("data/Noccaea_CQs.csv", stringsAsFactors = T)
input <- input[input$batch != "",] # Removes unprocessed (batch 3 + root rows)

input$rep <- as.factor(input$Biological.replicate)
input$genotype <- as.factor(input$Accession..)
phenotypes <- c("petiole", "margin", "vein", "tissue", "rand_5", "rand_10")
metals <- c("metal_Z", "metal_K", "metal_Ni", "metal_Ca")
metric <- "CQ"

for (metal in metals){
  for (current_phenotype in phenotypes){
    colname <- paste(metal, current_phenotype, metric, sep = "_")
    input$phenotype <- input[,colname]
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
    
    #mixed effects model H2 calculation 
    fitlmer_rand <- lmer(phenotype ~ (1|genotype), data = input, REML = TRUE) # + (1|row) + (1|column) if you needed 
    sum <- summary(fitlmer_rand)
    out <- as.data.frame(VarCorr(fitlmer_rand))
    if (current_phenotype == phenotypes[1]){
      result <- data.frame(out$vcov[1], out$vcov[2])
      names(result) <- c("genotype", "residual")
    } else {
      result[nrow(result) + 1,] = list(out$vcov[1], out$vcov[2])
    }
  }
  result$total_var <- result$genotype + result$residual
  result$H2 <- result$genotype / result$total_var
  result$H2_percent <- result$H2 * 100
  row.names(result) <- phenotypes
  
  print(
  ggplot(data=result, aes(x=row.names(result), y=H2_percent, fill=row.names(result))) +
    geom_bar(stat="identity") +
    #scale_fill_brewer(palette="Dark2") +
    ylim(0,100) +
    xlab("phenotype attributes") +
    ylab("H2 (%)") +
    theme(axis.text.x = element_text(angle = 90)) +
    # theme(legend.title=element_blank()) +
    theme(legend.position = "none") +
    ggtitle(metal)
    # scale_x_discrete(breaks=phenotypes,
    #                  labels=Plegend)
  # ggsave("data/output/rand_Z_H2.png") # Zimg_noise/H2.png
  )
  # scatter plots for one phenotype
  # sampled_accessions <- sample(input$Accession.., 15)
  # input_plot <- input[input$Accession.. %in% sampled_accessions,]
  # input_plot$Accession.. <- as.factor(input_plot$Accession..)
  # ggplot(input_plot, aes(x=Accession.., y=rand_Z_CQ_40, color=Accession..)) + geom_point()
  # 
  # sorted_result <- with(result,  result[order(row.names(result)) , ])
} 

