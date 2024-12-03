(require "datafusion.scm")
(require-builtin steel/ffi)

(define ctx (session-context))
; (define df (read-csv ctx "example.csv"))

(define (select df . cols)
  (df/select df cols))

;; Apply the UDF function here
(define (define/udf ctx name input-types output-type func)
  (define-udf ctx name input-types output-type (function->ffi-function func)))

; (define my-udf
;   (define-udf ctx
;               "custom-sum-function"
;               (list (Int64))
;               (Int64)
;               (function->ffi-function (lambda (col)
;                                         (displayln "Hello world!")
;                                         (arrow-sum (arrow-array-as-array-type col (Int64)))))))

(define df (read-csv ctx "pokemon.csv"))
;; Not exactly my favorite here, but we can get there
; (~> df
; (select (col "a") (col "b") (udf/call my-udf (list (col "a"))))
; (select (col/array-agg (col "Speed")))
; (df/filter (col>= (col "a") (col "b")))
; df/show)

(define type-1 (col "Type 1"))

(define (sort-by df . cols)
  (df/sort-by df cols))

(define (agg-builder-raw expr
                         #:distinct [distinct #f]
                         #:filter [filter #f]
                         #:order-by [order-by #f]
                         #:null-treatment [null-treatment #f])
  (agg/builder expr distinct filter order-by null-treatment))

(define (agg-builder func)
  (lambda (expr . args) (apply agg-builder-raw (cons (func expr) args))))

(define (agg-builder-2 func)
  (lambda (expr1 expr2 . args) (apply agg-builder-raw (cons (func expr1 expr2) args))))

(define array-agg-def (agg-builder col/array-agg))
(define count-def (agg-builder col/count))
(define max-def (agg-builder col/max))
(define min-def (agg-builder col/min))
(define avg-def (agg-builder col/avg))
(define median-def (agg-builder col/median))
(define stddev-def (agg-builder col/stddev))
(define stddev-pop-def (agg-builder col/stddev-pop))
(define var-sample-def (agg-builder col/var-sample))
(define var-pop-def (agg-builder col/var-pop))
(define approx-distinct-def (agg-builder col/approx-distinct))
(define approx-median-def (agg-builder col/approx-median))
(define regr-slope-def (agg-builder-2 col/regr-slope))
(define regr-intercept-def (agg-builder-2 col/regr-intercept))
(define regr-count-def (agg-builder-2 col/regr-count))
(define regr-r2-def (agg-builder-2 col/regr-r2))
(define regr-agvx-def (agg-builder-2 col/regr-avgx))
(define regr-avgy-def (agg-builder-2 col/regr-avgy))
(define regr-sxx-def (agg-builder-2 col/regr-sxx))
(define regr-syy-def (agg-builder-2 col/regr-syy))
(define regr-sxy-def (agg-builder-2 col/regr-sxy))

(~> df
    (df/filter (col/not-null? (col "Type 2")))
    (df/aggregate (list type-1)
                  (list (alias (array-agg-def (col "Type 2")
                                              #:distinct #true
                                              #:null-treatment (null-treatment-ignore-nulls))
                               "Type 2 List"))
                  ; (list (alias (col/count type-1) "Type 1 counts")
                  ;       (alias (col/avg (col "Speed")) "Average speed")
                  ;       (alias (col/avg (col "Attack")) "Average attack")
                  ;       ; (alias (col/avg (col "Defense")) "Average defense")
                  ;       )
                  )
    ; (sort-by (col "Type 1 counts"))
    df/show)
