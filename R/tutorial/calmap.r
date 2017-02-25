library(ggplot2)
library(latex2exp)
library(reshape2)

plot_calibration_map <- function(scores_set, info, legend_set, color_set, alpha=1){
  n_lines <- length(legend_set)
  sizes <- seq(1.5, 0.5, length.out = n_lines)
  bins <- seq(0, 1, length.out = 11)
  hist_tot <- hist(info$prob, breaks=bins, plot = FALSE)
  hist_pos <- hist(info$prob[info$labels == 1], breaks=bins, plot = FALSE)
  centers <- hist_tot$mids
  empirical <- (hist_pos$counts+alpha) / (hist_tot$counts+2*alpha)


  pdata <- melt(scores_set, id="linspace")
  i <- 1
  g <- ggplot(pdata, aes(x=linspace, y=value, colour=variable))
  for (legend in legend_set){
    g <- g + geom_line(size=sizes[i])
    i <- i + 1
  }

  df <- data.frame(centers, empirical)
  d <- melt(df, id="centers")
  g <- g + geom_point(data=d, aes(x=centers, y=value, colour=variable))


  g <- g + scale_colour_manual(values=c(color_set,'black'))
  g <- g + labs(x=TeX("$s$"),y=TeX("$\\hat{p}$"), title="Calibration map")
  g <- g + theme(plot.title = element_text(hjust = 0.5))
  g <- g + guides(colour = guide_legend("Method"))
  print(g)
}
