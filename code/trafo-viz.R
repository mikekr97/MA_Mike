# First thoughts on visualizing the transformation function
# For unconditional transformation models


# Dependencies ------------------------------------------------------------

library(tidyverse)
library(tram)
library(ggplot2)
library(ggpubr)
library(ordinal)

theme_set(theme_pubr())


################################## old , lucas #############################

# Data --------------------------------------------------------------------

data("faithful", package = "datasets")
data("wine", package = "ordinal")

# The model ---------------------------------------------------------------

m <- Colr(waiting ~ 1, data = faithful, order = 10, add = 2)

# Visualization -----------------------------------------------------------

setup <- function(object, res = 1e3, continuous = TRUE, nbin = 4, add = 1) {
  y <- variable.names(object, "response")
  qs <- mkgrid(object, n = res)[[y]]
  cdf <- predict(object, type = "distribution", q = qs)
  pdf <- predict(object, type = "density", q = qs)
  trafo <- predict(object, type = "trafo", q = qs)
  if (continuous) {
    pdfz <- object$model$todistr$d(trafo)
  } else {
    pdfz <- diff(c(0, object$model$todistr$p(trafo)))
  }
  dat <- data.frame(y = qs, cdf = cdf, pdf = pdf, trafo = trafo, pdfz = pdfz)
  if (is.numeric(qs))
    dat$cols <- cut(qs, quantile(qs, probs = seq(0, 1, length.out = nbin)))
  if (is.factor(qs)) {
    trafo2 <- seq(min(trafo) - add, max(trafo[is.finite(trafo)]) + add, 
               length.out = res)
    pdfz2 <- object$model$todistr$d(trafo2)
    cols2 <- cut(trafo2, breaks = c(-Inf, trafo))
    aux <- data.frame(trafo2 = trafo2, pdfz2 = pdfz2, cols2 = cols2)
    ret <- list(dat, aux)
    return(ret)
  }
  return(dat)
}

trafo_plot <- function(setup, continuous = TRUE, fill = FALSE, zlim = c(-4, 4),
                       tag = NULL) {

  p0 <- ggplot() + theme_void()

  if (!is.null(tag))
    p0 <- p0 + geom_text(aes(x = 0, y = 0, label = tag))

  if (continuous) {
    p1 <- ggplot(setup) +
      geom_line(aes(x = y, y = trafo, color = y)) +
      geom_segment(aes(x = y, xend = y, y = zlim[1], yend = trafo, color = y),
                   data = setup %>% group_by(cols) %>% filter(row_number() == 1), lty = 2) +
      geom_segment(aes(x = min(y), xend = y, y = trafo, yend = trafo, color = y),
                   data = setup %>% group_by(cols) %>% filter(row_number() == 1), lty = 2) +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.title.y = element_blank(), axis.ticks = element_blank(),
            axis.text = element_blank()) +
      scale_y_continuous(lim = zlim)

    p2 <- ggplot(setup) +
      geom_line(aes(x = trafo, y = pdfz, color = y)) +
      coord_flip() +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.x = element_blank()) +
      scale_x_continuous(lim = zlim) +
      labs(x = expression(h(y)))

    p3 <- ggplot(setup) +
      geom_line(aes(x = y, y = pdf, color = y)) +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.y = element_blank())
    p1 <- p1 + scale_color_viridis_c()
    p2 <- p2 + scale_color_viridis_c()
    p3 <- p3 + scale_color_viridis_c()
    if (fill) {
      p2 <- p2 + geom_area(aes(x = trafo, y = pdfz, fill = cols)) + scale_fill_viridis_d()
      p3 <- p3 + geom_area(aes(x = y, y = pdf, fill = cols)) + scale_fill_viridis_d()
    }
  } else {
    sup2 <- setup[[1]]
    sup2$trafo[!is.finite(sup2$trafo)] <- NA
    p1 <- ggplot(sup2) +
      geom_point(aes(x = y, y = trafo, color = y)) +
      geom_segment(aes(x = y, xend = y, y = zlim[1] - 0.5, yend = trafo, color = y),
                   data = sup2, lty = 2) +
      geom_segment(aes(x = as.numeric(min(y)) - 0.4, xend = y, y = trafo, yend = trafo, color = y),
                   data = sup2, lty = 2) +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.title.y = element_blank(), axis.ticks = element_blank(),
            axis.text = element_blank()) +
      scale_y_continuous(lim = range(setup[[2]]$trafo2))

    p2 <- ggplot(setup[[2]]) +
      geom_line(aes(x = trafo2, y = pdfz2)) +
      geom_area(aes(x = trafo2, y = pdfz2, fill = cols2), show.legend = FALSE) +
      coord_flip() +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.x = element_blank()) +
      scale_x_continuous(lim = range(setup[[2]]$trafo2)) +
      xlab("h(y)")

    p3 <- ggplot(setup[[1]]) +
      geom_segment(aes(x = y, xend = y, y = 0, yend = pdf, color = y)) +
      geom_point(aes(x = y, y = pdf, color = y)) +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.y = element_blank())
    p1 <- p1 + scale_fill_viridis_d() + coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
    p2 <- p2 + scale_fill_viridis_d()
    p3 <- p3 + scale_color_viridis_d() + coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
  }
  ggarrange(p2, p1, p0, p3, ncol = 2, nrow = 2)
}


# Continuous Model --------------------------------------------------------

data("faithful", package = "datasets")
m <- Colr(waiting ~ 1, data = faithful, order = 10, extrapolate = TRUE)
dat <- setup(m, nbin = 5)
(out <- trafo_plot(dat))
(outf <- trafo_plot(dat, fill = TRUE))

# Discrete Model ----------------------------------------------------------

data("wine", package = "ordinal")
m1 <- Polr(rating ~ 1, data = wine, method = "logistic")
dat1 <- setup(m1, continuous = FALSE, add = 2.5, nbin = 6)
(out1 <- trafo_plot(setup = dat1, continuous = FALSE))

# Discrete Model ----------------------------------------------------------

data("wine", package = "ordinal")
m2 <- Polr(rating ~ 1, data = wine, method = "cloglog")
dat2 <- setup(m2, continuous = FALSE, add = 2.5, nbin = 10)
(out2 <- trafo_plot(setup = dat2, continuous = FALSE))

# Save plots --------------------------------------------------------------

# ggsave(out, filename = "continuous.svg")
#ggsave(outf, filename = "continuous_filled.pdf")
#ggsave(out1, filename = "discrete.pdf")
# ggsave(out2, filename = "discrete-cloglog.pdf")

# Copenhagen talk ---------------------------------------------------------

df <- data.frame(y = 10 * rnorm(1e4) + 40)
m <- Lm(y ~ 1, data = df, extrapolate = TRUE)
dat <- setup(m, nbin = 10)
(outf <- trafo_plot(dat, fill = TRUE))
ggsave("nlrm.pdf", height = 4, width = 6)

data("wine", package = "ordinal")
m1 <- Polr(rating ~ 1, data = wine, method = "logistic")
dat1 <- setup(m1, continuous = FALSE, add = 2.5)
(out1 <- trafo_plot(setup = dat1, continuous = FALSE))
ggsave("polr.pdf", height = 4, width = 6)

data("faithful")
m1 <- Coxph(waiting ~ 1, data = faithful, order = 10, extrapolate = TRUE, 
            prob = c(0.1, 0.9), log_first = TRUE)
dat1 <- setup(m1, continuous = TRUE, add = 0)
(out1 <- trafo_plot(setup = dat1, fill = TRUE))
ggsave("coxph.pdf", height = 4, width = 6)

# library(ggside)
# smdat <- dat[seq(1, nrow(dat), 30),]
# dat %>% 
#   ggplot(aes(x = y, y = trafo, color = y)) +
#   geom_line() +
#   scale_color_viridis_c() +
#   geom_ysideline(aes(x = pdfz, y = trafo), orientation = "y") +
#   geom_xsideline(aes(x = y, y = pdf)) +
#   theme(ggside.panel.scale.x = 0.5, ggside.panel.scale.y = 0.5) +
#   geom_segment(aes(xend = max(y), yend = trafo), data = smdat) +
#   geom_segment(aes(xend = y, yend = max(trafo)), data = smdat) +
#   geom_line(aes(color = NULL)) +
#   geom_ysidesegment(aes(x = 0, xend = pdfz, y = trafo, yend = trafo), data = smdat) +
#   geom_xsidesegment(aes(x = y, xend = y, y = 0, yend = pdf), data = smdat) +
#   geom_xsidetext(aes(x = median(y), y = 1.2 * median(pdf), label = "f(y)"), inherit.aes = FALSE) +
#   geom_ysidetext(aes(x = median(pdfz), y = 0.5 * max(trafo), label = "f(z)"), inherit.aes = FALSE) +
#   theme(legend.position = "none")



###################### 1.11.24 ################## Causality Paper ##############

# 1. load packages at the beginning of script
# 2. execute all cells below -> plots for 2 continuous , 1 discrete
# 3. export figures as .svg
# 4. combine and change figures e.g. with pptx. -> https://medium.com/@buehler1991/the-visualization-manipulation-trick-8a4f5f6ddb3

# new trafo_plot with different colors
setup <- function(object, res = 1e3, continuous = TRUE, nbin = 4, add = 1) {
  y <- variable.names(object, "response")
  qs <- mkgrid(object, n = res)[[y]]
  cdf <- predict(object, type = "distribution", q = qs)
  pdf <- predict(object, type = "density", q = qs)
  trafo <- predict(object, type = "trafo", q = qs)
  if (continuous) {
    pdfz <- object$model$todistr$d(trafo)
  } else {
    pdfz <- diff(c(0, object$model$todistr$p(trafo)))
  }
  dat <- data.frame(y = qs, cdf = cdf, pdf = pdf, trafo = trafo, pdfz = pdfz)
  if (is.numeric(qs))
    dat$cols <- cut(qs, quantile(qs, probs = seq(0, 1, length.out = nbin)))
  if (is.factor(qs)) {
    trafo2 <- seq(min(trafo) - add, max(trafo[is.finite(trafo)]) + add, 
                  length.out = res)
    pdfz2 <- object$model$todistr$d(trafo2)
    cols2 <- cut(trafo2, breaks = c(-Inf, trafo))
    aux <- data.frame(trafo2 = trafo2, pdfz2 = pdfz2, cols2 = cols2)
    ret <- list(dat, aux)
    return(ret)
  }
  return(dat)
}
trafo_plot <- function(setup, continuous = TRUE, fill = FALSE, zlim = c(-4, 4),
                       tag = NULL, palette = "viridis") {
  
  p0 <- ggplot() + theme_void()
  
  if (!is.null(tag))
    p0 <- p0 + geom_text(aes(x = 0, y = 0, label = tag))
  
  if (continuous) {
    p1 <- ggplot(setup) +
      geom_line(aes(x = y, y = trafo, color = y)) +
      geom_segment(aes(x = y, xend = y, y = zlim[1], yend = trafo, color = y),
                   data = setup %>% group_by(cols) %>% filter(row_number() == 1), lty = 2) +
      geom_segment(aes(x = min(y), xend = y, y = trafo, yend = trafo, color = y),
                   data = setup %>% group_by(cols) %>% filter(row_number() == 1), lty = 2) +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.title.y = element_blank(), axis.ticks = element_blank(),
            axis.text = element_blank()) +
      scale_y_continuous(lim = zlim)
    
    p2 <- ggplot(setup) +
      geom_line(aes(x = trafo, y = pdfz, color = y)) +
      coord_flip() +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.x = element_blank()) +
      scale_x_continuous(lim = zlim) +
      labs(x = expression(h(y)))
    
    p3 <- ggplot(setup) +
      geom_line(aes(x = y, y = pdf, color = y)) +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.y = element_blank())
    
    # Apply viridis palettes with the appropriate option
    if (palette == "viridis") {
      p1 <- p1 + scale_color_viridis_c()
      p2 <- p2 + scale_color_viridis_c()
      p3 <- p3 + scale_color_viridis_c()
      if (fill) {
        p2 <- p2 + geom_area(aes(x = trafo, y = pdfz, fill = cols)) + scale_fill_viridis_d()
        p3 <- p3 + geom_area(aes(x = y, y = pdf, fill = cols)) + scale_fill_viridis_d()
      }
    } else if (palette == "plasma") {
      p1 <- p1 + scale_color_viridis_c(option = "plasma")
      p2 <- p2 + scale_color_viridis_c(option = "plasma")
      p3 <- p3 + scale_color_viridis_c(option = "plasma")
      if (fill) {
        p2 <- p2 + geom_area(aes(x = trafo, y = pdfz, fill = cols)) + scale_fill_viridis_d(option = "plasma")
        p3 <- p3 + geom_area(aes(x = y, y = pdf, fill = cols)) + scale_fill_viridis_d(option = "plasma")
      }
    } else {
      p1 <- p1 + scale_color_gradientn(colors = palette)
      p2 <- p2 + scale_color_gradientn(colors = palette)
      p3 <- p3 + scale_color_gradientn(colors = palette)
      if (fill) {
        p2 <- p2 + geom_area(aes(x = trafo, y = pdfz, fill = cols)) + scale_fill_gradientn(colors = palette)
        p3 <- p3 + geom_area(aes(x = y, y = pdf, fill = cols)) + scale_fill_gradientn(colors = palette)
      }
    }
    
  } else {
    sup2 <- setup[[1]]
    sup2$trafo[!is.finite(sup2$trafo)] <- NA
    p1 <- ggplot(sup2) +
      geom_point(aes(x = y, y = trafo, color = y)) +
      geom_segment(aes(x = y, xend = y, y = zlim[1] - 0.5, yend = trafo, color = y),
                   data = sup2, lty = 2) +
      geom_segment(aes(x = as.numeric(min(y)) - 0.4, xend = y, y = trafo, yend = trafo, color = y),
                   data = sup2, lty = 2) +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.title.y = element_blank(), axis.ticks = element_blank(),
            axis.text = element_blank()) +
      scale_y_continuous(lim = range(setup[[2]]$trafo2))
    
    p2 <- ggplot(setup[[2]]) +
      geom_line(aes(x = trafo2, y = pdfz2)) +
      geom_area(aes(x = trafo2, y = pdfz2, fill = cols2), show.legend = FALSE) +
      coord_flip() +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.x = element_blank()) +
      scale_x_continuous(lim = range(setup[[2]]$trafo2)) +
      xlab("h(y)")
    
    p3 <- ggplot(setup[[1]]) +
      geom_segment(aes(x = y, xend = y, y = 0, yend = pdf, color = y)) +
      geom_point(aes(x = y, y = pdf, color = y)) +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.y = element_blank())
    
    # Apply viridis palettes for discrete values
    if (palette == "viridis") {
      p1 <- p1 + scale_fill_viridis_d() + coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
      p2 <- p2 + scale_fill_viridis_d()
      p3 <- p3 + scale_color_viridis_d() + coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
    } else if (palette == "plasma") {
      p1 <- p1 + scale_fill_manual(values = viridisLite::plasma(n = length(unique(sup2$y)))) +
        coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
      p2 <- p2 + scale_fill_manual(values = viridisLite::plasma(n = length(unique(setup[[2]]$cols2))))
      p3 <- p3 + scale_color_manual(values = viridisLite::plasma(n = length(unique(sup2$y)))) +
        coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
    } else {
      p1 <- p1 + scale_fill_gradientn(colors = palette) + coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
      p2 <- p2 + scale_fill_gradientn(colors = palette)
      p3 <- p3 + scale_color_gradientn(colors = palette) + coord_trans(xlim = range(setup[[1]]$y) + c(-0.5, 0.5))
    }
  }
  ggarrange(p2, p1, p0, p3, ncol = 2, nrow = 2)
}

# X1 Continous
data("faithful", package = "datasets")
m <- Colr(waiting ~ 1, data = faithful, order = 10, extrapolate = TRUE)
dat <- setup(m, nbin = 10)
(out <- trafo_plot(dat))
(outf <- trafo_plot(dat, fill = TRUE,palette = 'plasma'))

# X2 Continous
data_bimodal <- c(rnorm(500,  50, 6), rnorm(500, 70,10))
data <- data.frame(waiting = data_bimodal)
m <- Colr(waiting ~ 1, data = data, order = 10, extrapolate = TRUE)
dat1 <- setup(m, nbin = 10)
(out <- trafo_plot(dat1))
(outf <- trafo_plot(dat1, fill = TRUE,palette = 'plasma'))

# X3 Discrete
data("wine", package = "ordinal")
wine_filtered <- subset(wine, rating < 5)
head(wine_filtered)
m2 <- Polr(rating ~ 1, data = wine_filtered, method = "logistic")
dat2 <- setup(m2, continuous = FALSE, add = 4, nbin = 3)
(out2 <- trafo_plot(setup = dat2, continuous = FALSE,palette = 'plasma'))




