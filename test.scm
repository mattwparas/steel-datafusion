(define ctx (session-context))
(define df (read-csv ctx "example.csv"))

(define (select df . cols)
  (df/select df cols))

(define my-udf
  (define-udf ctx "hello-world" (list Int64) Null (lambda (_) (displayln "Hello world!"))))

;; Not exactly my favorite here, but we can get there
(~> df
    (select (col "a") (col "b") (udf/call my-udf (list (col "a"))))
    (df/filter (col>= (col "a") (col "b")))
    df/show)
