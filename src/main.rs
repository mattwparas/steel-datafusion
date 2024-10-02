use std::sync::{Arc, Mutex};

use datafusion::{
    arrow::{array::RecordBatch, datatypes::DataType},
    common::JoinType,
    dataframe::DataFrame,
    execution::{context::SessionContext, options::CsvReadOptions},
    logical_expr::{col, create_udf, Expr, ScalarUDF, SimpleScalarUDF, SortExpr},
};
use steel::{
    rvals::{AsRefSteelVal as _, Custom, IntoSteelVal, SteelString},
    steel_vm::{builtin::BuiltInModule, engine::Engine, register_fn::RegisterFn, vm::VmContext},
    SteelErr, SteelVal,
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
struct SExpr(Expr);
impl Custom for SExpr {}

impl SExpr {
    fn col(name: String) -> Self {
        SExpr(col(name))
    }

    fn alias(self, name: String) -> Self {
        SExpr(self.0.alias(name))
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
}

struct SSessionContext(SessionContext);
impl Custom for SSessionContext {}

impl SSessionContext {
    fn new() -> Self {
        Self(SessionContext::new())
    }
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
        .register_fn("col", SExpr::col)
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
        "df/show",
        move |df: SDataFrame| -> Result<(), SDataFusionError> {
            rt.block_on(async { df.0.show().await })
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
    let func = args[2].clone();

    let rust_func = ctx_to_rust_function(ctx, func);

    let udf = create_udf(
        name,
        vec![DataType::Int64],
        Arc::new(datafusion::arrow::datatypes::DataType::Null),
        datafusion::logical_expr::Volatility::Immutable,
        Arc::new(move |_data| {
            println!("Calling...");
            (rust_func)(&mut []).unwrap();

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

    engine.register_module(datafusion_module());

    engine
        .run(
            r#"
            (require-builtin steel/datafusion)
        "#,
        )
        .unwrap();

    run_repl(engine).unwrap()
}
