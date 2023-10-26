#install.packages("PxWebApiData")
#install.packages("plotly")
library(PxWebApiData)
library(dplyr)
library(plotly)

# Locate the following dataset to use here https://data.ssb.no/api/v0/en/console

# First we download the norwegian GDP numbers 
gdp = ApiData("http://data.ssb.no/api/v0/en/table/09842", Tid = TRUE)
gdp = gdp[2] %>% as.data.frame()
gdp = gdp %>% filter(dataset.ContentsCode == "BNP")


electric = ApiData("http://data.ssb.no/api/v0/en/table/08307", returnMetaFrames = TRUE) 
electric[1] %>% as.data.frame()
electric[2] %>% as.data.frame()

el_dat = ApiData("http://data.ssb.no/api/v0/en/table/08307", Tid = TRUE)
data_c = el_dat[2] %>% as.data.frame()

el_dat = data_c %>% filter(dataset.ContentsCode == "Nettoforbruk" | dataset.ContentsCode == "ProdTotal")

y = el_dat %>% filter(dataset.ContentsCode == "Nettoforbruk") %>% select(dataset.value, dataset.Tid)
x = el_dat %>% filter(dataset.ContentsCode == "ProdTotal") %>% select(dataset.value, dataset.Tid)
merged = inner_join(y, x, by = "dataset.Tid")
colnames(merged) = c("pow_con", "year", "pow_prod")

#formating_table
working_df <- data.frame(
  year = gdp$dataset.Tid,
  bnp = gdp$dataset.value
)

working_df$year = working_df$year %>% as.numeric()
merged$year = merged$year %>% as.numeric()

merged_dataset <- inner_join(merged, working_df, by = "year")

merged_dataset <- merged_dataset %>%
  select(year, everything())

########## A bit of data exploration ####################

summary(merged_dataset)

correlation_gdp_pcons <- cor(merged_dataset$bnp, merged_dataset$pow_con)
correlation_gdp_pprod <- cor(merged_dataset$bnp, merged_dataset$pow_prod)

# Use Fisher’s Z-transformation to compare correlations
z1 <- 0.5 * log((1 + correlation_gdp_pcons) / (1 - correlation_gdp_pcons))
z2 <- 0.5 * log((1 + correlation_gdp_pprod) / (1 - correlation_gdp_pprod))
se <- sqrt(1 / (nrow(merged_dataset) - 3)) #

z <- (z1 - z2) / se


plot = plot_ly(merged_dataset, x = ~year)
plot <- plot %>% add_trace(y = ~(bnp), name = 'bnp',mode = 'lines') 
plot <- plot %>% add_trace(y = ~(pow_con), name = 'GhW Nettoforbruk Strøm',mode = 'lines') 
plot <- plot %>% add_trace(y = ~(pow_prod), name = "GhW Total Strøm produksjon", mode = 'lines')
plot

plot = NULL


############### Lin reg ###########
model <- lm(bnp ~ pow_con + pow_prod, data = merged_dataset)
summary(model)

par(mfrow=c(2,2))
plot(model)
