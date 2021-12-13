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

use std::sync::Arc;

use anyhow::Result;

fn load_gfa(gfa_path: &str) -> Result<(PackedGraph, PathPositionMap)> {
    let mut mmap = gfa::mmap::MmapGFA::new(gfa_path)?;
    let graph = gfa2obj::load::packed_graph_from_mmap(&mut mmap)?;
    let path_positions = PathPositionMap::index_paths(&graph);
    Ok((graph, path_positions))
}

fn main() {
    println!("Hello, world!");
}
