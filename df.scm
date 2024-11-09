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

(define (agg-builder expr
                     #:distinct [distinct #f]
                     #:filter [filter #f]
                     #:order-by [order-by #f]
                     #:null-treatment [null-treatment #f])
  (agg/builder expr distinct filter order-by null-treatment))

(~> df
    (df/filter (col/not-null? (col "Type 2")))
    (df/aggregate (list type-1)
                  (list (alias (agg-builder (col/array-agg (col "Type 2"))
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
