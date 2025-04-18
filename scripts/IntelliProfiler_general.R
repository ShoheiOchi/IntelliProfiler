#Checking packages
mypkg <- "tidyverse"
if(require(mypkg, character.only=TRUE)){
  print("tidyverse is loaded correctly")
} else {
  print("trying to install tidyverse...")
  install.packages(tidyverse)
  if(require(tidyverse)){
    print("tidyverse installed and loaded")
  } else {
    stop("could not install tidyverse")
  }
}
if(require(openxlsx)){
  print("openxlsx is loaded correctly")
} else {
  print("trying to install openxlsx")
  install.packages(openxlsx)
  if(require("openxlsx")){
    print("openxlsx installed and loaded")
  } else {
    stop("could not install openxlsx")
  }
}
library(lubridate)


# Import and format data set
# Select your file
file_path <- file.choose()  # Opens a dialog to select the file
df <- read.table(file_path, header = FALSE, sep = "]")
colnames(df) <- c("Date", "PositionID")
df$Date <- gsub(df$Date, pattern="\\[", replacement = "")
df <- df %>% separate(PositionID, c("Position", "ID"), sep=":")
df$Position <- as.integer(df$Position)
df$ID <- gsub(df$ID, pattern=",", replacement = "_")
df$Date <- as.POSIXct(df$Date, format="%Y-%m-%d %H:%M:%OS")
df$Date <- floor_date(df$Date, unit = "second")


# Prepare a template data frame for date
time_start <- df$Date[1]
time_end <- df$Date[nrow(df)]
date_seq <- seq(from = time_start, to = time_end, by = "sec")  
date_df <- data.frame(date_seq)
colnames(date_df) <- "Date"


# Prepare Position and X-Y coordinates
list_positon <-  seq(1, 96) 
list_X <- rep(seq(1, 12, 1), each = 4)
list_X <- append(list_X, rep(seq(6, 1, -1), each = 4))
list_X <- append(list_X, rep(seq(12, 7, -1), each = 4))
list_Y <- rep(seq(4, 1, -1), 12)
list_Y <- append(list_Y, rep(seq(5, 8, 1), 12))

X <- as.vector(list_X[match(df$Position, list_positon)])
Y <- as.vector(list_Y[match(df$Position, list_positon)])


# Add the X-Y coordinates to data frame
df$X <- X
df$Y <- Y


# Extract transponder ID and number
list_ID <- strsplit(df$ID, "_")
list_ID <- unlist(list_ID)
list_ID <- unique(list_ID)
number_transponder = length(list_ID)  #check number of transponder

# Calculate euclidean distance from t to t+1
euc_dist <- function(df) {
  a <- c(df["x1"], df["y1"])
  b <- c(df["x2"], df["y2"])
  return (sqrt(sum((a - b)^2)))
}


# Split data to each ID, delete duplicate, fill NA with values.
for (i in 1:number_transponder) {
  tmp <- filter(df, str_detect(ID, list_ID[i]))
  tmp <- tmp %>% distinct(Date, .keep_all=T)
  tmp <- merge(date_df, tmp, by = "Date", all = TRUE)
  tmp <- fill(tmp, c(Position, ID, X, Y), .direction = "down")
  # Calculate travel distance
  t <- tmp
  t <- t[-nrow(t),]
  t_plus_1 <- tmp
  t_plus_1 <- t_plus_1[-1,]
  dist_df <-data.frame(t$X, t$Y, t_plus_1$X, t_plus_1$Y)
  colnames(dist_df) <- c("x1", "y1", "x2", "y2")
  tmp_list <- apply(dist_df, 1, euc_dist)
  tmp_list <- append(c(NA), tmp_list)
  tmp$dist <- tmp_list
  assign(list_ID[i], tmp)
  # Save data frame as an excel file
  base_name <- tools::file_path_sans_ext(basename(file_path)) 
  f_name <- str_c(base_name, "_", list_ID[i], ".xlsx")  
  write.xlsx(get(list_ID[i]), f_name)
}


# Set start point
start_time <- date_df$Date[1] 
#start_time <- as.POSIXct("2022-01-27 19:00:00", format="%Y-%m-%d %H:%M:%OS") 
#Set duration by sec
duration_time <- 3600 
# Set end point
#end_time <- date_df$Date[nrow(date_df)]
end_time <- start_time + duration_time
#end_time <- as.POSIXct("2022-01-27 19:00:00", format="%Y-%m-%d %H:%M:%OS") 

# Get start and end point index
s = grep(start_time, date_df$Date)
e = grep(end_time, date_df$Date)


for (i in 1:number_transponder) {
  #2D plot
  name_ID = str_c(list_ID[i])  #store transponder ID as string
  f_name <- str_c("2D_plot_", base_name, "_", list_ID[i], ".pdf")  
  pdf(f_name, width=12, height=8)
  plot(get(list_ID[i])$X[s:e], get(list_ID[i])$Y[s:e], type="l", col=i,
       xlab="X_position", ylab="Y_position", main = name_ID,
       xlim=c(1, 12), xaxp=c(1, 12, 11), ylim=c(1, 8), yaxp=c(1, 8, 7))
  dev.off() 
}

for (i in 1:number_transponder) {
  #Time series of X and Y
  name_ID = str_c(list_ID[i])  #store transponder ID as string
  f_name <- str_c("X-Y_plot_", list_ID[i], ".pdf")
  pdf(f_name, width=12, height=6)
  plot(get(list_ID[i])$Date[s:e], get(list_ID[i])$X[s:e], type="l", col="blue",
       xlab="Time", ylab="Position", main = name_ID,
       ylim=c(1, 12), yaxp=c(1, 12, 11))
  par(new=T)
  plot(get(list_ID[i])$Date[s:e], get(list_ID[i])$Y[s:e], type="l", col="red",
       ylim=c(1, 12), yaxp=c(1, 12, 11), ann=F)
  legend("topright", legend=c("X_position", "Y_position"), lty=1, col=c("blue","red"))
  dev.off() 
}

for (i in 1:(number_transponder-1)) {
  for (j in (i+1):number_transponder) {
    m <- get(list_ID[i])
    n <- get(list_ID[j])
    dist_df2 <-data.frame(m$X, m$Y, n$X, n$Y)
    colnames(dist_df2) <- c("x1", "y1", "x2", "y2")
    tmp_list2 <- apply(dist_df2, 1, euc_dist)
    
    name_ID = str_c(list_ID[i], "_vs_",  list_ID[j]) #store transponder ID as string
    f_name <- str_c("Distance_plot_", name_ID, ".pdf")
    pdf(f_name, width=12, height=6)
    plot(get(list_ID[i])$Date[s:e], tmp_list2[s:e], type="l", col=i,
         xlab="Time", ylab="Position", main = name_ID,
         ylim=c(1, 12), yaxp=c(1, 12, 11))
    dev.off()
  }
}


# Travel distance
for (i in 1:number_transponder) {
  name_ID = str_c(list_ID[i])  #store transponder ID as string
  f_name <- str_c("Travel_distance_plot_", list_ID[i], ".pdf")
  pdf(f_name, width=12, height=6)
  plot(get(list_ID[i])$Date[s:e], get(list_ID[i])$dist[s:e], type="l", col=i,
       xlab="Time", ylab="Travel distance", main = name_ID,
       ylim=c(1, 12), yaxp=c(1, 12, 11))
  dev.off() 
}



