---
title: "Test chunks"
author: "Test author"
format: html
---


``` r
library(ggsql)
test_file <- "test.csv"
write.csv(mtcars, test_file)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2-1.png)


``` r
unlink(test_file)
```
