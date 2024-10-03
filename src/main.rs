use std::sync::{Arc, Mutex};

use datafusion::{
    arrow::{array::RecordBatch, datatypes::DataType},
    common::JoinType,
    dataframe::DataFrame,
    execution::{context::SessionContext, options::CsvReadOptions},
    logical_expr::{
        case, col, conditional_expressions::CaseBuilder, create_udf, when, Expr, ScalarUDF,
        SortExpr,
    },
};
use steel::{
    rvals::{AsRefSteelVal as _, Custom, FromSteelVal as _, IntoSteelVal, SteelString},
    steel_vm::{builtin::BuiltInModule, engine::Engine, register_fn::RegisterFn},
    stop, SteelErr, SteelVal,
};
use steel_repl::run_repl;

use datafusion::common::DataFusionError;

#[derive(Clone)]
struct SDataFrame(DataFrame);
impl Custom for SDataFrame {}

pub struct SDataFusionError(DataFusionError);
impl Custom for SDataFusionError {
    fn fmt(&self) -> Option<std::result::Result<String, std::fmt::Error>> {
        Some(Ok(self.0.to_string()))
    }
}

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

    fn with_column(self, name: SteelString, expr: SExpr) -> Result<Self, SDataFusionError> {
        self.0
            .with_column(name.as_str(), expr.0)
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }

    fn with_column_renamed(
        self,
        old_name: String,
        new_name: SteelString,
    ) -> Result<Self, SDataFusionError> {
        self.0
            .with_column_renamed(old_name, new_name.as_str())
            .map(SDataFrame)
            .map_err(SDataFusionError)
    }
}

fn col_add(expr: &[SteelVal]) -> Result<SteelVal, SteelErr> {
    if expr.len() == 0 {
        stop!(ArityMismatch => "col+ expects at least one column");
    }

    let mut col_iter = expr.into_iter().map(SExpr::as_ref);

    let mut init = col_iter.next().unwrap()?.clone().0;

    for value in col_iter {
        let value = value?;

        init = init.clone() + value.clone().0;
    }

    SExpr(init).into_steelval()
}

fn col_sub(expr: &[SteelVal]) -> Result<SteelVal, SteelErr> {
    if expr.len() == 0 {
        stop!(ArityMismatch => "col- expects at least one column");
    }

    let mut col_iter = expr.into_iter().map(SExpr::as_ref);

    let mut init = col_iter.next().unwrap()?.clone().0;

    for value in col_iter {
        let value = value?;

        init = init.clone() - value.clone().0;
    }

    SExpr(init).into_steelval()
}

fn col_multiply(expr: &[SteelVal]) -> Result<SteelVal, SteelErr> {
    if expr.len() == 0 {
        stop!(ArityMismatch => "col* expects at least one column");
    }

    let mut col_iter = expr.into_iter().map(SExpr::as_ref);

    let mut init = col_iter.next().unwrap()?.clone().0;

    for value in col_iter {
        let value = value?;

        init = init.clone() * value.clone().0;
    }

    SExpr(init).into_steelval()
}

fn col_divide(expr: &[SteelVal]) -> Result<SteelVal, SteelErr> {
    if expr.len() == 0 {
        stop!(ArityMismatch => "col/ expects at least one column");
    }

    let mut col_iter = expr.into_iter().map(SExpr::as_ref);

    let mut init = col_iter.next().unwrap()?.clone().0;

    for value in col_iter {
        let value = value?;

        init = init.clone() / value.clone().0;
    }

    SExpr(init).into_steelval()
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

fn datafusion_data_types() -> BuiltInModule {
    let mut module = BuiltInModule::new("steel/datafusion/datatypes");

    // TODO: Include more datatypes
    module
        .register_value(
            "Null",
            ArrowDataType(DataType::Null).into_steelval().unwrap(),
        )
        .register_value(
            "Boolean",
            ArrowDataType(DataType::Boolean).into_steelval().unwrap(),
        )
        .register_value(
            "Int8",
            ArrowDataType(DataType::Int8).into_steelval().unwrap(),
        )
        .register_value(
            "Int16",
            ArrowDataType(DataType::Int16).into_steelval().unwrap(),
        )
        .register_value(
            "Int32",
            ArrowDataType(DataType::Int32).into_steelval().unwrap(),
        )
        .register_value(
            "Int64",
            ArrowDataType(DataType::Int64).into_steelval().unwrap(),
        )
        .register_value(
            "UInt8",
            ArrowDataType(DataType::UInt8).into_steelval().unwrap(),
        )
        .register_value(
            "UInt16",
            ArrowDataType(DataType::UInt16).into_steelval().unwrap(),
        )
        .register_value(
            "UInt32",
            ArrowDataType(DataType::UInt32).into_steelval().unwrap(),
        )
        .register_value(
            "UInt64",
            ArrowDataType(DataType::UInt64).into_steelval().unwrap(),
        )
        .register_value(
            "Float16",
            ArrowDataType(DataType::Float16).into_steelval().unwrap(),
        )
        .register_value(
            "Float32",
            ArrowDataType(DataType::Float32).into_steelval().unwrap(),
        )
        .register_value(
            "Float64",
            ArrowDataType(DataType::Float64).into_steelval().unwrap(),
        )
        .register_value(
            "Binary",
            ArrowDataType(DataType::Binary).into_steelval().unwrap(),
        )
        .register_value(
            "LargeBinary",
            ArrowDataType(DataType::LargeBinary)
                .into_steelval()
                .unwrap(),
        )
        .register_value(
            "Utf8",
            ArrowDataType(DataType::Utf8).into_steelval().unwrap(),
        )
        .register_value(
            "LargeUtf8",
            ArrowDataType(DataType::LargeUtf8).into_steelval().unwrap(),
        );

    module
}

fn datafusion_module() -> BuiltInModule {
    let mut module = BuiltInModule::new("steel/datafusion");

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
        .register_value("col+", SteelVal::FuncV(col_add))
        .register_value("col-", SteelVal::FuncV(col_sub))
        .register_value("col*", SteelVal::FuncV(col_multiply))
        .register_value("col/", SteelVal::FuncV(col_divide))
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
        move |ctx: &SSessionContext, path: SteelString| -> Result<SDataFrame, SDataFusionError> {
            rt.block_on(async { ctx.0.read_csv(path.as_str(), CsvReadOptions::new()).await })
                .map(SDataFrame)
                .map_err(SDataFusionError)
        },
    );

    module.register_value("define-udf", SteelVal::BuiltIn(define_udf));

    module
}

fn ctx_to_rust_function(
    ctx: &steel::steel_vm::vm::VmCore,
    func: SteelVal,
) -> Box<dyn Fn(&mut [SteelVal]) -> Result<SteelVal, SteelErr> + Send + Sync + 'static> {
    let thread = Arc::new(Mutex::new(ctx.make_thread()));

    Box::new(move |args: &mut [SteelVal]| {
        let mut guard = thread.lock().unwrap();
        let map = guard.get_constant_map();
        guard.call_function_from_mut_slice(map, func.clone(), args)
    })
}

pub(crate) fn define_udf(
    ctx: &mut steel::steel_vm::vm::VmCore,
    args: &[SteelVal],
) -> Option<Result<SteelVal, SteelErr>> {
    let session_ctx = SSessionContext::as_ref(&args[0]).unwrap();
    let name = args[1].as_string().unwrap();

    let types = <Vec<ArrowDataType>>::from_steelval(&args[2])
        .unwrap()
        .into_iter()
        .map(|x| x.0)
        .collect::<Vec<_>>();

    let return_type = ArrowDataType::from_steelval(&args[3]).unwrap();

    let func = args[4].clone();

    let rust_func = ctx_to_rust_function(ctx, func);

    let udf = create_udf(
        name,
        types,
        Arc::new(return_type.0),
        datafusion::logical_expr::Volatility::Immutable,
        Arc::new(move |_data| {
            // TODO: How to do zero copy of the underlying types?
            (rust_func)(&mut [SteelVal::Void]).unwrap();

            Ok(datafusion::logical_expr::ColumnarValue::Scalar(
                datafusion::scalar::ScalarValue::Null,
            ))
        }),
    );

    // Register the UDF so that we can... use it?
    session_ctx.0.register_udf(udf.clone());

    Some(SteelScalarUDF(udf).into_steelval())
}

fn main() {
    let mut engine = Engine::new();

    engine
        .register_module(datafusion_module())
        .register_module(datafusion_data_types());

    engine
        .run(
            r#"
            (require-builtin steel/datafusion)
            (require-builtin steel/datafusion/datatypes)
        "#,
        )
        .unwrap();

    run_repl(engine).unwrap()
}
