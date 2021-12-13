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
use na::{right_handed, Vec2, Vec3};
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

use std::sync::Arc;

use anyhow::Result;
use argh::FromArgs;

// use nalgebra as na;
use nalgebra_glm as na;

#[derive(Debug, FromArgs)]
/// gfa2obj
pub struct Args {
    /// the GFA file to load
    #[argh(positional)]
    gfa_path: String,
}

// pub enum WorldNode {
//     Left(NodeId),
//     Right(NodeId),
// }

#[derive(Clone)]
pub struct Chain {
    left: Handle,
    right: Handle,

    contained_nodes: FxHashSet<NodeId>,

    length: usize,
    // children: Vec<Arc<Mutex<Option<Chain>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathRange {
    path: PathId,
    start: StepPtr,
    end: StepPtr,

    start_h: Handle,
    end_h: Handle,
}

impl PathRange {
    pub fn from_path(graph: &PackedGraph, path: PathId) -> Option<Self> {
        let start = graph.path_first_step(path)?;
        let end = graph.path_last_step(path)?;

        let start_h = graph.path_handle_at_step(path, start)?;
        let end_h = graph.path_handle_at_step(path, end)?;

        Some(Self {
            path,
            start,
            end,

            start_h,
            end_h,
        })
    }

    // pub fn remove_
}

#[derive(Default, Clone)]
pub struct Layout {
    path_1d: Path1DLayout,

    nodes: Vec<(Vec3, Vec3)>,

    vertices: Vec<Vec3>,
    links: Vec<usize>,
}

impl Layout {
    // pub fn from_gfa<T: OptFields>(gfa: &GFA<usize, T>) -> Self {
    pub fn from_graph(graph: &PackedGraph) -> Self {
        let layout_1d = Path1DLayout::new(graph);

        let zero: Vec3 = na::vec3(0.0, 0.0, 0.0);

        // graph.path_handle_at_step(id, index)

        let longest_path = graph
            .path_ids()
            .max_by_key(|path| layout_1d.path_len(*path).unwrap_or(0))
            .unwrap();

        let mut covered: FxHashSet<(usize, usize)> = FxHashSet::default();

        let mut x: f32 = 0.0;

        let scale = 10.0;
        let padding = 1.0;

        let mut nodes = vec![(zero, zero); graph.node_count()];

        {
            let mut append_node = |id: NodeId| {
                let ranges = layout_1d.path_ranges.get(&longest_path).unwrap();
                let len = graph.node_len(Handle::pack(id, false));
                let ix = (id.0 - 1) as usize;
                let x0 = x;
                let x1 = x + (len as f32) / scale;
                x = x1 + padding;
                nodes[ix] = (na::vec3(x0, 0.0, 0.0), na::vec3(x1, 0.0, 0.0));
            };

            let steps = graph.path_steps(longest_path).unwrap();

            for step in steps {
                let h = step.handle();
                let node = step.handle().id();
                // append_node(step.handle().id());
            }
        }

        unimplemented!();
    }
}

// pub struct NodePositions {
// }

#[derive(Debug, Default, Clone)]
pub struct Obj {
    vertices: Vec<(f32, f32)>,
    links: Vec<(usize, usize)>,
}

fn load_gfa(gfa_path: &str) -> Result<(PackedGraph, PathPositionMap)> {
    let mut mmap = gfa::mmap::MmapGFA::new(gfa_path)?;
    let graph = gfa2obj::load::packed_graph_from_mmap(&mut mmap)?;
    let path_positions = PathPositionMap::index_paths(&graph);
    Ok((graph, path_positions))
}

pub const BP_PER_UNIT: usize = 1_000;

pub fn len_to_pos(l: usize) -> f32 {
    (l as f32) / (BP_PER_UNIT as f32)
}

pub struct VecChain {
    vertices: Vec<na::Vec3>,
    links: Vec<(usize, usize)>,

    start: na::Vec3,
    end: na::Vec3,
}

impl VecChain {
    pub fn print_obj(&self) {
        let mut min_s = std::usize::MAX;
        let mut max_s = std::usize::MIN;

        let mut min_e = std::usize::MAX;
        let mut max_e = std::usize::MIN;

        for vx in self.vertices.iter() {
            println!("v {} {} {}", vx.x, vx.y, vx.z);
        }

        for (s, e) in self.links.iter() {
            // .obj uses 1-based indices
            let s = s + 1;
            let e = e + 1;

            min_s = min_s.min(s);
            max_s = max_s.max(s);
            min_e = min_e.min(e);
            max_e = max_e.max(e);
            println!("l {} {}", s, e);
        }

        eprintln!("vertices.len(): {}", self.vertices.len());
        eprintln!("links.len(): {}", self.links.len());

        eprintln!("s, min: {}\tmax: {}", min_s, max_s);
        eprintln!("e, min: {}\tmax: {}", min_e, max_e);
    }

    pub fn from_path(graph: &PackedGraph, path: PathId) -> Option<Self> {
        let mut steps = graph.path_steps(path)?;

        let first = steps.next()?;

        let h0 = first.handle();

        let node_len = graph.node_len(h0);
        let start = na::vec3(0.0, 0.0, 0.0);

        let end = na::vec3(len_to_pos(node_len), 0.0, 0.0);

        let mut chain = VecChain {
            // vertices: vec![start, end],
            // links: vec![(0, 1)],
            vertices: vec![start],
            links: vec![],
            start,
            end: start,
        };

        for step in steps {
            let h = step.handle();
            // let node = step.handle().id();
            let node_len = graph.node_len(h);
            chain.append_node(node_len);
            // append_node(step.handle().id());
        }

        Some(chain)
    }

    pub fn append_node(&mut self, node_len: usize) {
        let ix0 = self.vertices.len() - 1;
        let ix1 = ix0 + 1;

        let vx0 = self.end;

        let delta = na::vec3(len_to_pos(node_len), 0.0, 0.0);

        let vx1 = vx0 + delta;

        self.vertices.push(vx1);
        self.links.push((ix0, ix1));

        self.end = vx1;
    }
}

fn main() {
    let args: Args = argh::from_env();

    let (graph, path_pos) = load_gfa(&args.gfa_path).unwrap();

    let layout_1d = Path1DLayout::new(&graph);

    let longest_path = graph
        .path_ids()
        .max_by_key(|path| layout_1d.path_len(*path).unwrap_or(0))
        .unwrap();

    let chain = VecChain::from_path(&graph, longest_path).unwrap();

    chain.print_obj();

    /*
    let mut used_nodes: FxHashSet<NodeId> = FxHashSet::default();

    let mut remaining_len = graph.total_length();

    let steps = graph.path_steps(longest_path).unwrap();

    // let mut

    for step in steps {
        let h = step.handle();
        let node = step.handle().id();

        let node_len = graph.node_len(h);

        used_nodes.insert(node);
        remaining_len -= node_len;
        // append_node(step.handle().id());
    }

    println!("total     len: {}", graph.total_length());
    println!("remaining len: {}", remaining_len);
    */

    // let mut layout = Layout::from_graph(&graph);

    // let mut node_pos: FxHashMap<NodeId, (Vec2, Vec2)> = FxHashMap::default();
}

#[derive(Debug, Clone, Default)]
pub struct Path1DLayout {
    pub total_len: usize,
    pub path_ranges: FxHashMap<PathId, Vec<std::ops::Range<usize>>>,
    node_offsets: Vec<usize>,
}

impl Path1DLayout {
    fn path_len(&self, path: PathId) -> Option<usize> {
        let ranges = self.path_ranges.get(&path)?;
        Some(ranges.iter().map(|range| range.end - range.start).sum())
    }

    fn node_at_global(&self, pos: usize) -> Option<NodeId> {
        let ix = self.node_offsets.binary_search(&pos);

        match ix {
            Ok(ix) => Some(NodeId::from((ix + 1) as u64)),
            Err(ix) => {
                let ix = ix + 1;
                // let _ = self.node_offsets.get(&ix)?;
                if ix >= self.node_offsets.len() {
                    return None;
                }
                Some(NodeId::from((ix + 1) as u64))
            }
        }
    }

    fn new(graph: &PackedGraph) -> Self {
        let nodes = {
            let mut ns = graph.handles().map(|h| h.id()).collect::<Vec<_>>();
            ns.sort();
            ns
        };

        let path_count = graph.path_count();

        let mut open_ranges: Vec<Option<usize>> = vec![None; path_count];
        let mut path_ranges: Vec<Vec<std::ops::Range<usize>>> = vec![Vec::new(); path_count];
        let mut total_len = 0usize;
        let mut paths_on_handle = FxHashSet::default();
        let mut node_offsets = Vec::with_capacity(nodes.len());

        for node in nodes {
            let handle = Handle::pack(node, false);

            let len = graph.node_len(handle);

            node_offsets.push(total_len);

            let x0 = total_len;

            paths_on_handle.clear();
            paths_on_handle.extend(
                graph
                    .steps_on_handle(handle)
                    .into_iter()
                    .flatten()
                    .map(|(path, _)| path),
            );

            for (ix, (open, past)) in open_ranges
                .iter_mut()
                .zip(path_ranges.iter_mut())
                .enumerate()
            {
                let path = PathId(ix as u64);

                let offset = x0;
                let on_path = paths_on_handle.contains(&path);

                if let Some(s) = open {
                    if !on_path {
                        past.push(*s..offset);
                        *open = None;
                    }
                } else {
                    if on_path {
                        *open = Some(offset);
                    }
                }
            }

            total_len += len;
        }

        for (open, past) in open_ranges.iter_mut().zip(path_ranges.iter_mut()) {
            if let Some(s) = open {
                past.push(*s..total_len);
                *open = None;
            }
        }

        let path_ranges: FxHashMap<PathId, Vec<std::ops::Range<usize>>> = path_ranges
            .into_iter()
            .enumerate()
            .map(|(ix, ranges)| (PathId(ix as u64), ranges))
            .collect();

        Self {
            total_len,
            path_ranges,
            node_offsets,
        }
    }
}
