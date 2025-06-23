library(ggplot2)

# Generate data
set.seed(42)
x <- seq(1, 10, length.out = 300)
y <- 2 * x + rnorm(300, sd = 1)
df <- data.frame(x = x, y = y)

# Fit linear model
model <- lm(y ~ x, data = df)

# Select x-values where to show conditional distributions
x_vals <- seq(2, 8, by = 2)

# Create conditional distribution data
cond_dists_list <- lapply(x_vals, function(x0) {
	y_seq <- seq(predict(model, data.frame(x = x0)) - 4 * summary(model)$sigma,
							 predict(model, data.frame(x = x0)) + 4 * summary(model)$sigma, length.out = 200)
	density_vals <- dnorm(y_seq, mean = predict(model, newdata = data.frame(x = x0)),
												sd = summary(model)$sigma)
	data.frame(
		x0 = x0,
		y = y_seq,
		x = x0 + density_vals * 1.5 # Adjust scaling for better visual width
	)
})
cond_dists <- do.call(rbind, cond_dists_list)

# Plot
ggplot(df, aes(x, y)) +
	geom_point(alpha = 0.2, size = 0.6) +
	geom_abline(intercept = coef(model)[1], slope = coef(model)[2], color = "black", size = 1) +
	geom_path(data = cond_dists, aes(x = x, y = y, group = x0), color = "blue", size = 1) +
	theme_minimal() +
	labs(x = "X", y = "Y") +
	theme(
		axis.title = element_text(size = 12),
		axis.text = element_text(size = 10),
		axis.ticks = element_line(),
		panel.grid = element_blank(),
		# axis.line = element_line(color = "black", linewidth = 0.5) # Add axis lines,
		axis.line = element_line(arrow = arrow(length = unit(0.2, "cm"), ends = "last"))
	)
# Save the plot




