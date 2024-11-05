(require "datafusion.scm")
(require-builtin steel/ffi)

(define ctx (session-context))
(define df (read-csv ctx "example.csv"))

(define (select df . cols)
  (df/select df cols))

(define my-udf
  (define-udf ctx
              "custom-max-function"
              (list (Int64))
              (Int64)
              (function->ffi-function (lambda (col)
                                        (displayln "Hello world!")
                                        (arrow-max (arrow-array-as-array-type col (Int64)))))))

;; Not exactly my favorite here, but we can get there
(~> df
    (select (col "a") (col "b") (udf/call my-udf (list (col "a"))))
    (df/filter (col>= (col "a") (col "b")))
    df/show)
