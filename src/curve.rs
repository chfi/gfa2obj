use gfa::{gfa::GFA, optfields::OptFields};
#[allow(unused_imports)]
use handlegraph::{
    handle::{Direction, Handle, NodeId},
    handlegraph::*,
    mutablehandlegraph::*,
    packed::*,
    pathhandlegraph::*,
};

use handlegraph::{
    packedgraph::{paths::StepPtr, PackedGraph},
    path_position::PathPositionMap,
};

use crossbeam::channel;
use na::{Vec2, Vec3};
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

use std::sync::Arc;

use anyhow::Result;
use argh::FromArgs;

// use nalgebra as na;
use nalgebra_glm as na;

pub struct CurveDecomp {
    // top_level: Chain,
    top_level: (usize, usize),
}

#[derive(Clone)]
pub struct Chain {
    // vx_range: std::ops::Range<usize>,
    path: PathId,
    start: StepPtr,
    end: StepPtr,

    // start_h: Handle,
    // end_h: Handle,
    nodes: Vec<NodeId>,

    children: FxHashMap<(usize, usize), Chain>,
}
