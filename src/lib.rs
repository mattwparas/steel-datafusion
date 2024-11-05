use std::{error::Error, sync::Arc};

use abi_stable::std_types::{RBoxError, RSliceMut};
use datafusion::{
    arrow::{
        array::{
            ArrowPrimitiveType, BooleanArray, Date32Array, DurationNanosecondArray, Float16Array,
            Float64Array, PrimitiveArray, RecordBatch,
        },
        compute,
        datatypes::{
            DataType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
            DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
            DurationSecondType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type,
            Int64Type, Int8Type, IntervalDayTimeType, IntervalMonthDayNanoType,
            IntervalYearMonthType, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
            Time64NanosecondType, TimestampMicrosecondType, TimestampMillisecondType,
            TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt64Type,
            UInt8Type,
        },
    },
    common::JoinType,
    dataframe::DataFrame,
    execution::{context::SessionContext, options::CsvReadOptions},
    logical_expr::{
        case, col, conditional_expressions::CaseBuilder, create_udf, when, ColumnarValue, Expr,
        ScalarUDF, SortExpr,
    },
};
use steel::{
    rvals::Custom,
    steel_vm::ffi::{FFIModule, FFIValue, HostRuntimeFunction, IntoFFIVal, RegisterFFIFn},
};

use datafusion::common::DataFusionError;

#[derive(Clone)]
struct SDataFrame(DataFrame);
impl Custom for SDataFrame {}

#[derive(Debug)]
pub struct SDataFusionError(DataFusionError);
impl Custom for SDataFusionError {
    fn fmt(&self) -> Option<std::result::Result<String, std::fmt::Error>> {
        Some(Ok(self.0.to_string()))
    }
}

impl std::fmt::Display for SDataFusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for SDataFusionError {}

#[derive(Clone)]
pub struct SExpr(Expr);
impl Custom for SExpr {}

impl SExpr {
    fn col(name: String) -> Self {
        SExpr(col(name))
    }

    fn alias(self, name: String) -> Self {
        SExpr(self.0.alias(name))
    }

    fn and(left: SExpr, right: SExpr) -> Self {
        SExpr(datafusion::logical_expr::and(left.0, right.0))
    }

    fn or(left: SExpr, right: SExpr) -> Self {
        SExpr(datafusion::logical_expr::or(left.0, right.0))
    }

    fn lt_eq(self, other: SExpr) -> SExpr {
        SExpr(self.0.lt_eq(other.0))
    }

    fn gt_eq(self, other: SExpr) -> SExpr {
        SExpr(self.0.gt_eq(other.0))
    }

    fn gt(self, other: SExpr) -> SExpr {
        SExpr(self.0.gt(other.0))
    }

    fn lt(self, other: SExpr) -> SExpr {
        SExpr(self.0.gt(other.0))
    }

    fn eq(self, other: SExpr) -> SExpr {
        SExpr(self.0.eq(other.0))
    }

    fn not_eq(self, other: SExpr) -> SExpr {
        SExpr(self.0.not_eq(other.0))
    }

    fn like(self, other: SExpr) -> SExpr {
        SExpr(self.0.like(other.0))
    }

    fn ilike(self, other: SExpr) -> SExpr {
        SExpr(self.0.ilike(other.0))
    }

    fn not_like(self, other: SExpr) -> SExpr {
        SExpr(self.0.not_like(other.0))
    }

    fn not_ilike(self, other: SExpr) -> SExpr {
        SExpr(self.0.not_ilike(other.0))
    }
}

#[derive(Clone)]
struct SSortExpr(SortExpr);
impl Custom for SSortExpr {}

#[derive(Clone)]
struct SJoinType(JoinType);
impl Custom for SJoinType {}

#[derive(Clone)]
pub struct SRecordBatch(RecordBatch);
impl Custom for SRecordBatch {}

pub struct SteelScalarUDF(ScalarUDF);
impl Custom for SteelScalarUDF {}

pub struct SCaseBuilder(CaseBuilder);
impl Custom for SCaseBuilder {}

impl SCaseBuilder {
    pub fn case(expr: SExpr) -> Self {
        SCaseBuilder(case(expr.0))
    }

    pub fn when(when_expr: SExpr, then: SExpr) -> Self {
        SCaseBuilder(when(when_expr.0, then.0))
    }

    fn add_when(&mut self, when_expr: SExpr, then: SExpr) -> Self {
        SCaseBuilder(self.0.when(when_expr.0, then.0))
    }

    fn otherwise(&mut self, otherwise: SExpr) -> Result<SExpr, SDataFusionError> {
        self.0
            .otherwise(otherwise.0)
            .map(SExpr)
            .map_err(SDataFusionError)
    }

    pub fn end(&self) -> Result<SExpr, SDataFusionError> {
        self.0.end().map(SExpr).map_err(SDataFusionError)
    }
}

impl SteelScalarUDF {
    fn call(&self, args: Vec<SExpr>) -> SExpr {
        SExpr(self.0.call(args.into_iter().map(|x| x.0).collect()))
    }
}

impl SDataFrame {
    fn union(self, df: SDataFrame) -> Result<SDataFrame, SDataFusionError> {
        self.0.union(df.0).map(SDataFrame).map_err(SDataFusionError)
    }

    fn union_distinct(self, df: SDataFrame) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .union_distinct(df.0)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn distinct(self) -> Result<SDataFrame, SDataFusionError> {
        self.0.distinct().map(SDataFrame).map_err(SDataFusionError)
    }

    fn distinct_on(
        self,
        on_expr: Vec<SExpr>,
        select_expr: Vec<SExpr>,
        sort_expr: Option<Vec<SSortExpr>>,
    ) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .distinct_on(
                on_expr.into_iter().map(|x| x.0).collect(),
                select_expr.into_iter().map(|x| x.0).collect(),
                sort_expr.map(|x| x.into_iter().map(|x| x.0).collect()),
            )
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn filter(self, predicate: SExpr) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .filter(predicate.0)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn aggregate(
        self,
        group_expr: Vec<SExpr>,
        aggr_expr: Vec<SExpr>,
    ) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .aggregate(
                group_expr.into_iter().map(|x| x.0).collect(),
                aggr_expr.into_iter().map(|x| x.0).collect(),
            )
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn window(self, window_exprs: Vec<SExpr>) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .window(window_exprs.into_iter().map(|x| x.0).collect())
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn limit(self, skip: usize, fetch: Option<usize>) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .limit(skip, fetch)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn sort_by(self, expr: Vec<SExpr>) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .sort_by(expr.into_iter().map(|x| x.0).collect())
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn sort(self, expr: Vec<SSortExpr>) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .sort(expr.into_iter().map(|x| x.0).collect())
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn select(self, expr_list: Vec<SExpr>) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .select(expr_list.into_iter().map(|x| x.0).collect())
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn join(
        self,
        right: SDataFrame,
        join_type: SJoinType,
        left_cols: Vec<String>,
        right_cols: Vec<String>,
        filter: Option<SExpr>,
    ) -> Result<SDataFrame, SDataFusionError> {
        let left_cols: Vec<&str> = left_cols.iter().map(|x| x.as_str()).collect();
        let right_cols: Vec<&str> = right_cols.iter().map(|x| x.as_str()).collect();

        self.0
            .join(
                right.0,
                join_type.0,
                &left_cols,
                &right_cols,
                filter.map(|x| x.0),
            )
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn join_on(
        self,
        right: SDataFrame,
        join_type: SJoinType,
        on_exprs: Vec<SExpr>,
    ) -> Result<SDataFrame, SDataFusionError> {
        self.0
            .join_on(right.0, join_type.0, on_exprs.into_iter().map(|x| x.0))
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn explain(self, verbose: bool, analyze: bool) -> Result<Self, SDataFusionError> {
        self.0
            .explain(verbose, analyze)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn intersect(self, dataframe: SDataFrame) -> Result<Self, SDataFusionError> {
        self.0
            .intersect(dataframe.0)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn except(self, dataframe: SDataFrame) -> Result<Self, SDataFusionError> {
        self.0
            .except(dataframe.0)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn with_column(self, name: String, expr: SExpr) -> Result<Self, SDataFusionError> {
        self.0
            .with_column(name.as_str(), expr.0)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn with_column_renamed(
        self,
        old_name: String,
        new_name: String,
    ) -> Result<Self, SDataFusionError> {
        self.0
            .with_column_renamed(old_name, new_name.as_str())
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }
}

fn col_add(exprs: Vec<SExpr>) -> Result<SExpr, SDataFusionError> {
    if exprs.len() == 0 {
        let converted: Box<dyn Error + Send + Sync> =
            "col+ expects at least one column".to_string().into();
        return Err(SDataFusionError(DataFusionError::External(converted)));
    }

    let mut col_iter = exprs.into_iter();
    let mut init = col_iter.next().unwrap().clone().0;

    for value in col_iter {
        init = init.clone() + value.clone().0;
    }

    Ok(SExpr(init))
}

fn col_sub(exprs: Vec<SExpr>) -> Result<SExpr, SDataFusionError> {
    if exprs.len() == 0 {
        let converted: Box<dyn Error + Send + Sync> =
            "col- expects at least one column".to_string().into();
        return Err(SDataFusionError(DataFusionError::External(converted)));
    }

    let mut col_iter = exprs.into_iter();
    let mut init = col_iter.next().unwrap().clone().0;

    for value in col_iter {
        init = init.clone() - value.clone().0;
    }

    Ok(SExpr(init))
}

fn col_multiply(exprs: Vec<SExpr>) -> Result<SExpr, SDataFusionError> {
    if exprs.len() == 0 {
        let converted: Box<dyn Error + Send + Sync> =
            "col* expects at least one column".to_string().into();
        return Err(SDataFusionError(DataFusionError::External(converted)));
    }

    let mut col_iter = exprs.into_iter();
    let mut init = col_iter.next().unwrap().clone().0;

    for value in col_iter {
        init = init.clone() * value.clone().0;
    }

    Ok(SExpr(init))
}

fn col_divide(exprs: Vec<SExpr>) -> Result<SExpr, SDataFusionError> {
    if exprs.len() == 0 {
        let converted: Box<dyn Error + Send + Sync> =
            "col/ expects at least one column".to_string().into();
        return Err(SDataFusionError(DataFusionError::External(converted)));
    }

    let mut col_iter = exprs.into_iter();
    let mut init = col_iter.next().unwrap().clone().0;

    for value in col_iter {
        init = init.clone() / value.clone().0;
    }

    Ok(SExpr(init))
}

struct SSessionContext(SessionContext);
impl Custom for SSessionContext {}

impl SSessionContext {
    fn new() -> Self {
        Self(SessionContext::new())
    }
}

#[derive(Clone)]
struct ArrowDataType(DataType);
impl Custom for ArrowDataType {}

fn datafusion_module() -> FFIModule {
    let mut module = FFIModule::new("steel/datafusion");

    // Just use this to block on things until I figure out a better way
    // to embed the runtime.
    let runtime = Arc::new(tokio::runtime::Runtime::new().unwrap());

    module
        .register_fn("df/union", SDataFrame::union)
        .register_fn("df/union-distinct", SDataFrame::union_distinct)
        .register_fn("df/distinct", SDataFrame::distinct)
        .register_fn("df/distinct-on", SDataFrame::distinct_on)
        .register_fn("df/filter", SDataFrame::filter)
        .register_fn("df/aggregate", SDataFrame::aggregate)
        .register_fn("df/window", SDataFrame::window)
        .register_fn("df/limit", SDataFrame::limit)
        .register_fn("df/sort-by", SDataFrame::sort_by)
        .register_fn("df/sort", SDataFrame::sort)
        .register_fn("df/select", SDataFrame::select)
        .register_fn("df/join", SDataFrame::join)
        .register_fn("df/join-on", SDataFrame::join_on)
        .register_fn("df/explain", SDataFrame::explain)
        .register_fn("df/intersect", SDataFrame::intersect)
        .register_fn("df/except", SDataFrame::except)
        .register_fn("df/with-column", SDataFrame::with_column)
        .register_fn("df/with-column-renamed", SDataFrame::with_column_renamed)
        .register_fn("col", SExpr::col)
        .register_fn("col+", col_add)
        .register_fn("col-", col_sub)
        .register_fn("col*", col_multiply)
        .register_fn("col/", col_divide)
        .register_fn("col/and", SExpr::and)
        .register_fn("col/or", SExpr::or)
        .register_fn("col>=", SExpr::gt_eq)
        .register_fn("col<=", SExpr::lt_eq)
        .register_fn("col>", SExpr::gt)
        .register_fn("col<", SExpr::lt)
        .register_fn("col=", SExpr::eq)
        .register_fn("col!=", SExpr::not_eq)
        .register_fn("col/like", SExpr::like)
        .register_fn("col/ilike", SExpr::ilike)
        .register_fn("col/not-like", SExpr::not_like)
        .register_fn("col/not-ilike", SExpr::not_ilike)
        .register_fn("col/case", SCaseBuilder::case)
        .register_fn("case/when", SCaseBuilder::when)
        .register_fn("case/end", SCaseBuilder::end)
        .register_fn("case/with-when", SCaseBuilder::add_when)
        .register_fn("case/otherwise", SCaseBuilder::otherwise)
        .register_fn("alias", SExpr::alias)
        .register_fn("session-context", SSessionContext::new)
        .register_fn("udf/call", SteelScalarUDF::call);

    let rt = runtime.clone();
    module.register_fn(
        "df/collect",
        move |df: SDataFrame| -> Result<Vec<SRecordBatch>, SDataFusionError> {
            rt.block_on(async { df.0.collect().await })
                .map(|x| x.into_iter().map(SRecordBatch).collect())
                .map_err(SDataFusionError)
        },
    );

    let rt = runtime.clone();
    module.register_fn(
        "df/describe",
        move |df: SDataFrame| -> Result<SDataFrame, SDataFusionError> {
            rt.block_on(async { df.0.describe().await })
                .map(SDataFrame)
                .map_err(SDataFusionError)
        },
    );

    let rt = runtime.clone();
    module.register_fn(
        "df/count",
        move |df: SDataFrame| -> Result<usize, SDataFusionError> {
            rt.block_on(async { df.0.count().await })
                .map_err(SDataFusionError)
        },
    );

    let rt = runtime.clone();
    module.register_fn(
        "df/show",
        move |df: SDataFrame| -> Result<(), SDataFusionError> {
            rt.block_on(async { df.0.show().await })
                .map_err(SDataFusionError)
        },
    );

    let rt = runtime.clone();
    module.register_fn(
        "df/show-limit",
        move |df: SDataFrame, num: usize| -> Result<(), SDataFusionError> {
            rt.block_on(async { df.0.show_limit(num).await })
                .map_err(SDataFusionError)
        },
    );

    let rt = runtime.clone();
    module.register_fn(
        "read-csv",
        move |ctx: &SSessionContext, path: String| -> Result<SDataFrame, SDataFusionError> {
            rt.block_on(async { ctx.0.read_csv(path.as_str(), CsvReadOptions::new()).await })
                .map(SDataFrame)
                .map_err(SDataFusionError)
        },
    );

    module
        .register_value(
            "Null",
            ArrowDataType(DataType::Null).into_ffi_val().unwrap(),
        )
        .register_value(
            "Boolean",
            ArrowDataType(DataType::Boolean).into_ffi_val().unwrap(),
        )
        .register_value(
            "Int8",
            ArrowDataType(DataType::Int8).into_ffi_val().unwrap(),
        )
        .register_value(
            "Int16",
            ArrowDataType(DataType::Int16).into_ffi_val().unwrap(),
        )
        .register_value(
            "Int32",
            ArrowDataType(DataType::Int32).into_ffi_val().unwrap(),
        )
        .register_value(
            "Int64",
            ArrowDataType(DataType::Int64).into_ffi_val().unwrap(),
        )
        .register_value(
            "UInt8",
            ArrowDataType(DataType::UInt8).into_ffi_val().unwrap(),
        )
        .register_value(
            "UInt16",
            ArrowDataType(DataType::UInt16).into_ffi_val().unwrap(),
        )
        .register_value(
            "UInt32",
            ArrowDataType(DataType::UInt32).into_ffi_val().unwrap(),
        )
        .register_value(
            "UInt64",
            ArrowDataType(DataType::UInt64).into_ffi_val().unwrap(),
        )
        .register_value(
            "Float16",
            ArrowDataType(DataType::Float16).into_ffi_val().unwrap(),
        )
        .register_value(
            "Float32",
            ArrowDataType(DataType::Float32).into_ffi_val().unwrap(),
        )
        .register_value(
            "Float64",
            ArrowDataType(DataType::Float64).into_ffi_val().unwrap(),
        )
        .register_value(
            "Binary",
            ArrowDataType(DataType::Binary).into_ffi_val().unwrap(),
        )
        .register_value(
            "LargeBinary",
            ArrowDataType(DataType::LargeBinary).into_ffi_val().unwrap(),
        )
        .register_value(
            "Utf8",
            ArrowDataType(DataType::Utf8).into_ffi_val().unwrap(),
        )
        .register_value(
            "LargeUtf8",
            ArrowDataType(DataType::LargeUtf8).into_ffi_val().unwrap(),
        );

    module.register_fn("define-udf", define_udf);

    // Try this?
    module
        .register_fn("arrow-and", arrow_and)
        .register_fn("arrow-and-kleene", arrow_and_kleene)
        .register_fn("arrow-of", arrow_or)
        .register_fn("arrow-and-not", arrow_and_not)
        .register_fn("arrow-not", arrow_not)
        .register_fn("arrow-max", arrow_max)
        .register_fn("arrow-min", arrow_min)
        .register_fn("arrow-sum", arrow_sum);

    module
}

steel::declare_module!(build_module);

pub fn build_module() -> FFIModule {
    datafusion_module()
}

struct SColumnarValue(ColumnarValue);
impl Custom for SColumnarValue {}

fn define_udf(
    session_ctx: &SSessionContext,
    name: String,
    types: Vec<ArrowDataType>,
    return_type: ArrowDataType,
    func: HostRuntimeFunction,
) -> SteelScalarUDF {
    let udf = create_udf(
        &name,
        types.into_iter().map(|x| x.0).collect(),
        Arc::new(return_type.0),
        datafusion::logical_expr::Volatility::Immutable,
        Arc::new(move |args| {
            let first = &args[0];

            if let ColumnarValue::Array(a) = first {
                let base = a
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("cast failed");

                todo!()
            }

            func.call(RSliceMut::from_mut_slice(&mut [FFIValue::Void]))
                .unwrap();

            Ok(datafusion::logical_expr::ColumnarValue::Scalar(
                datafusion::scalar::ScalarValue::Null,
            ))
        }),
    );
    // Register the UDF so that we can... use it?
    session_ctx.0.register_udf(udf.clone());

    SteelScalarUDF(udf)
}

pub struct SPrimitiveArray<T: ArrowPrimitiveType>(PrimitiveArray<T>);

impl<T: ArrowPrimitiveType> Custom for SPrimitiveArray<T> {}

pub type SDate32Array = SPrimitiveArray<Date32Type>;
pub type SDate64Array = SPrimitiveArray<Date64Type>;
pub type SDecimal128Array = SPrimitiveArray<Decimal128Type>;
pub type SDecimal256Array = SPrimitiveArray<Decimal256Type>;
pub type SDurationMicrosecondArray = SPrimitiveArray<DurationMicrosecondType>;
pub type SDurationMillisecondArray = SPrimitiveArray<DurationMillisecondType>;
pub type SDurationNanosecondArray = SPrimitiveArray<DurationNanosecondType>;
pub type SDurationSecondArray = SPrimitiveArray<DurationSecondType>;
pub type SFloat16Array = SPrimitiveArray<Float16Type>;
pub type SFloat32Array = SPrimitiveArray<Float32Type>;
pub type SFloat64Array = SPrimitiveArray<Float64Type>;
pub type SInt8Array = SPrimitiveArray<Int8Type>;
pub type SInt16Array = SPrimitiveArray<Int16Type>;
pub type SInt32Array = SPrimitiveArray<Int32Type>;
pub type SInt64Array = SPrimitiveArray<Int64Type>;
pub type SIntervalDayTimeArray = SPrimitiveArray<IntervalDayTimeType>;
pub type SIntervalMonthDayNanoArray = SPrimitiveArray<IntervalMonthDayNanoType>;
pub type SIntervalYearMonthArray = SPrimitiveArray<IntervalYearMonthType>;
pub type STime32MillisecondArray = SPrimitiveArray<Time32MillisecondType>;
pub type STime32SecondArray = SPrimitiveArray<Time32SecondType>;
pub type STime64MicrosecondArray = SPrimitiveArray<Time64MicrosecondType>;
pub type STime64NanosecondArray = SPrimitiveArray<Time64NanosecondType>;
pub type STimestampMicrosecondArray = SPrimitiveArray<TimestampMicrosecondType>;
pub type STimestampMillisecondArray = SPrimitiveArray<TimestampMillisecondType>;
pub type STimestampNanosecondArray = SPrimitiveArray<TimestampNanosecondType>;
pub type STimestampSecondArray = SPrimitiveArray<TimestampSecondType>;
pub type SUInt8Array = SPrimitiveArray<UInt8Type>;
pub type SUInt16Array = SPrimitiveArray<UInt16Type>;
pub type SUInt32Array = SPrimitiveArray<UInt32Type>;
pub type SUInt64Array = SPrimitiveArray<UInt64Type>;

pub enum ArrowPrimitiveValue {
    Date32(<Date32Type as ArrowPrimitiveType>::Native),
    Date64(<Date64Type as ArrowPrimitiveType>::Native),
    Decimal128(<Decimal128Type as ArrowPrimitiveType>::Native),
    Decimal256(<Decimal256Type as ArrowPrimitiveType>::Native),
    DurationMicrosecond(<DurationMicrosecondType as ArrowPrimitiveType>::Native),
    DurationMillisecond(<DurationMillisecondType as ArrowPrimitiveType>::Native),
    DurationNanosecond(<DurationNanosecondType as ArrowPrimitiveType>::Native),
    DurationSecond(<DurationSecondType as ArrowPrimitiveType>::Native),
    Float16(<Float16Type as ArrowPrimitiveType>::Native),
    Float32(<Float32Type as ArrowPrimitiveType>::Native),
    Float64(<Float64Type as ArrowPrimitiveType>::Native),
    Int8(<Int8Type as ArrowPrimitiveType>::Native),
    Int16(<Int16Type as ArrowPrimitiveType>::Native),
    Int32(<Int32Type as ArrowPrimitiveType>::Native),
    Int64(<Int64Type as ArrowPrimitiveType>::Native),
    IntervalDayTime(<IntervalDayTimeType as ArrowPrimitiveType>::Native),
    IntervalMonthDayNano(<IntervalMonthDayNanoType as ArrowPrimitiveType>::Native),
    IntervalYearMonth(<IntervalYearMonthType as ArrowPrimitiveType>::Native),
    Time32Millisecond(<Time32MillisecondType as ArrowPrimitiveType>::Native),
    Time32Second(<Time32SecondType as ArrowPrimitiveType>::Native),
    Time64Microsecond(<Time64MicrosecondType as ArrowPrimitiveType>::Native),
    Time64Nanosecond(<Time64NanosecondType as ArrowPrimitiveType>::Native),
    TimestampMicrosecond(<TimestampMicrosecondType as ArrowPrimitiveType>::Native),
    TimestampMillisecond(<TimestampMillisecondType as ArrowPrimitiveType>::Native),
    TimestampNanosecond(<TimestampNanosecondType as ArrowPrimitiveType>::Native),
    TimestampSecond(<TimestampSecondType as ArrowPrimitiveType>::Native),
    UInt8(<UInt8Type as ArrowPrimitiveType>::Native),
    UInt16(<UInt16Type as ArrowPrimitiveType>::Native),
    UInt32(<UInt32Type as ArrowPrimitiveType>::Native),
    UInt64(<UInt64Type as ArrowPrimitiveType>::Native),
}

impl Custom for ArrowPrimitiveValue {}

pub enum SPrimitiveArrayKind {
    Date32(SDate32Array),
    Date64(SDate64Array),
    Decimal128Array(SDecimal128Array),
    Decimal256Array(SDecimal256Array),
    DurationMicrosecondArray(SDurationMicrosecondArray),
    DurationMillisecondArray(SDurationMillisecondArray),
    DurationNanosecondArray(SDurationNanosecondArray),
    DurationSecondArray(SDurationSecondArray),
    Float16Array(SFloat16Array),
    Float32Array(SFloat32Array),
    Float64Array(SFloat64Array),
    Int8Array(SInt8Array),
    Int16Array(SInt16Array),
    Int32Array(SInt32Array),
    Int64Array(SInt64Array),
    IntervalDayTimeArray(SIntervalDayTimeArray),
    IntervalMonthDayNanoArray(SIntervalMonthDayNanoArray),
    IntervalYearMonthArray(SIntervalYearMonthArray),
    Time32MillisecondArray(STime32MillisecondArray),
    Time32SecondArray(STime32SecondArray),
    Time64MicrosecondArray(STime64MicrosecondArray),
    Time64NanosecondArray(STime64NanosecondArray),
    TimestampMicrosecondArray(STimestampMicrosecondArray),
    TimestampMillisecondArray(STimestampMillisecondArray),
    TimestampNanosecondArray(STimestampNanosecondArray),
    TimestampSecondArray(STimestampSecondArray),
    UInt8Array(SUInt8Array),
    UInt16Array(SUInt16Array),
    UInt32Array(SUInt32Array),
    UInt64Array(SUInt64Array),
}

impl Custom for SPrimitiveArrayKind {}

#[derive(Clone)]
pub struct SBooleanArray(BooleanArray);
impl Custom for SBooleanArray {}

// TODO: Implement bindings to these functions:
// https://docs.rs/arrow/latest/arrow/compute/index.html

fn arrow_and(
    left: &SBooleanArray,
    right: SBooleanArray,
) -> Result<SBooleanArray, SDataFusionError> {
    compute::and(&left.0, &right.0)
        .map_err(|e| SDataFusionError(DataFusionError::ArrowError(e, None)))
        .map(SBooleanArray)
}

fn arrow_and_kleene(
    left: &SBooleanArray,
    right: SBooleanArray,
) -> Result<SBooleanArray, SDataFusionError> {
    compute::and_kleene(&left.0, &right.0)
        .map_err(|e| SDataFusionError(DataFusionError::ArrowError(e, None)))
        .map(SBooleanArray)
}

fn arrow_or(left: &SBooleanArray, right: SBooleanArray) -> Result<SBooleanArray, SDataFusionError> {
    compute::or(&left.0, &right.0)
        .map_err(|e| SDataFusionError(DataFusionError::ArrowError(e, None)))
        .map(SBooleanArray)
}

fn arrow_and_not(
    left: &SBooleanArray,
    right: SBooleanArray,
) -> Result<SBooleanArray, SDataFusionError> {
    compute::and_not(&left.0, &right.0)
        .map_err(|e| SDataFusionError(DataFusionError::ArrowError(e, None)))
        .map(SBooleanArray)
}

fn arrow_not(left: &SBooleanArray) -> Result<SBooleanArray, SDataFusionError> {
    compute::not(&left.0)
        .map_err(|e| SDataFusionError(DataFusionError::ArrowError(e, None)))
        .map(SBooleanArray)
}

macro_rules! arrow_function_map {
    ($name:tt, $func:expr) => {
        fn $name(kind: &SPrimitiveArrayKind) -> Option<ArrowPrimitiveValue> {
            match kind {
                SPrimitiveArrayKind::Date32(a) => $func(&a.0).map(ArrowPrimitiveValue::Date32),
                SPrimitiveArrayKind::Date64(a) => $func(&a.0).map(ArrowPrimitiveValue::Date64),
                SPrimitiveArrayKind::Decimal128Array(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Decimal128)
                }
                SPrimitiveArrayKind::Decimal256Array(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Decimal256)
                }
                SPrimitiveArrayKind::DurationMicrosecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::DurationMicrosecond)
                }
                SPrimitiveArrayKind::DurationMillisecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::DurationMillisecond)
                }
                SPrimitiveArrayKind::DurationNanosecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::DurationNanosecond)
                }
                SPrimitiveArrayKind::DurationSecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::DurationSecond)
                }
                SPrimitiveArrayKind::Float16Array(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Float16)
                }
                SPrimitiveArrayKind::Float32Array(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Float32)
                }
                SPrimitiveArrayKind::Float64Array(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Float64)
                }
                SPrimitiveArrayKind::Int8Array(a) => $func(&a.0).map(ArrowPrimitiveValue::Int8),
                SPrimitiveArrayKind::Int16Array(a) => $func(&a.0).map(ArrowPrimitiveValue::Int16),
                SPrimitiveArrayKind::Int32Array(a) => $func(&a.0).map(ArrowPrimitiveValue::Int32),
                SPrimitiveArrayKind::Int64Array(a) => $func(&a.0).map(ArrowPrimitiveValue::Int64),
                SPrimitiveArrayKind::IntervalDayTimeArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::IntervalDayTime)
                }
                SPrimitiveArrayKind::IntervalMonthDayNanoArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::IntervalMonthDayNano)
                }
                SPrimitiveArrayKind::IntervalYearMonthArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::IntervalYearMonth)
                }
                SPrimitiveArrayKind::Time32MillisecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Time32Millisecond)
                }
                SPrimitiveArrayKind::Time32SecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Time32Second)
                }
                SPrimitiveArrayKind::Time64MicrosecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Time64Microsecond)
                }
                SPrimitiveArrayKind::Time64NanosecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::Time64Nanosecond)
                }
                SPrimitiveArrayKind::TimestampMicrosecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::TimestampMicrosecond)
                }
                SPrimitiveArrayKind::TimestampMillisecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::TimestampMillisecond)
                }
                SPrimitiveArrayKind::TimestampNanosecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::TimestampNanosecond)
                }
                SPrimitiveArrayKind::TimestampSecondArray(a) => {
                    $func(&a.0).map(ArrowPrimitiveValue::TimestampSecond)
                }
                SPrimitiveArrayKind::UInt8Array(a) => $func(&a.0).map(ArrowPrimitiveValue::UInt8),
                SPrimitiveArrayKind::UInt16Array(a) => $func(&a.0).map(ArrowPrimitiveValue::UInt16),
                SPrimitiveArrayKind::UInt32Array(a) => $func(&a.0).map(ArrowPrimitiveValue::UInt32),
                SPrimitiveArrayKind::UInt64Array(a) => $func(&a.0).map(ArrowPrimitiveValue::UInt64),
            }
        }
    };
}

arrow_function_map!(arrow_max, compute::max);
arrow_function_map!(arrow_min, compute::min);
arrow_function_map!(arrow_sum, compute::sum);
