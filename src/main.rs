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

use bstr::ByteSlice;

#[derive(Debug, FromArgs)]
/// gfa2obj
pub struct Args {
    /// the GFA file to load
    #[argh(positional)]
    gfa_path: String,
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

#[derive(Default, Clone)]
pub struct Layout3D {
    handle_vx: FxHashMap<Handle, usize>,

    vertices: Vec<na::Vec3>,
    links: FxHashMap<usize, usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ChainIx {
    chain_id: usize,
    offset: usize,
}

#[derive(Default, Clone)]
pub struct Chains {
    // root is assumed to be index 0
    chains: Vec<Vec<Handle>>,
    structure: Vec<FxHashMap<(Handle, Handle), usize>>,

    inv_map: FxHashMap<Handle, ChainIx>,
}

impl Chains {
    pub fn len(&self) -> usize {
        self.chains.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    pub fn push_chain(&mut self, in_chain: &[Handle]) -> usize {
        // should we handle pre-placed handles here? probs not
        // i'll just assume the input is correct

        let ix = self.push_empty();

        let chain = &mut self.chains[ix];

        for &handle in in_chain {
            chain.push(handle);
        }

        ix
    }

    pub fn push_empty(&mut self) -> usize {
        let ix = self.chains.len();

        self.chains.push(Vec::new());
        self.structure.push(FxHashMap::default());

        ix
    }
}

// impl Layout3D {
//     pub fn add_chain(&mut self, chain: &[Handle]) {
// for (h, _, p) in chain {
// let len =
// }
//
//     }
// }

/*
pub struct Layout3D {
    vertices: Vec<na::Vec3>,
    // links: Vec<(usize, usize)>,
    link_map: FxHashMap<usize, usize>,

    // each node is represented by a sequence of vertices
    node_vx_map: FxHashMap<NodeId, std::ops::Range<usize>>,

    top_level: Chain,

    chain_vx_map: FxHashMap<(PathId, StepPtr, StepPtr), std::ops::Range<usize>>,
}

impl Layout3D {

    /// Start a layout using a single path as the spine
    pub fn from_path(graph: &PackedGraph, path: PathId) -> Option<Self> {
        let mut steps = graph.path_steps(path)?;

        let first = steps.next()?;

        let h0 = first.handle();

        let node_len = graph.node_len(h0);
        let start = na::vec3(0.0, 0.0, 0.0);

        let end = na::vec3(len_to_pos(node_len), 0.0, 0.0);

        unimplemented!();
    }
}
*/

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

        let mut added_nodes: FxHashSet<NodeId> = FxHashSet::default();

        for step in steps {
            let h = step.handle();
            let node = step.handle().id();
            if !added_nodes.insert(node) {
                continue;
            }

            let node_len = graph.node_len(h);
            chain.append_node(node_len);
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

    let path_name = graph.get_path_name_vec(longest_path).unwrap();

    println!("using longest path: {}", path_name.as_bstr());

    let mut chain_map = ChainMap::new(&graph, &path_pos);
    let longest_chain = chain_map
        .remove_path(longest_path)
        .unwrap()
        .remaining
        .remove(0);

    let mut remaining_nodes =
        graph.handles().map(|h| h.id()).collect::<FxHashSet<_>>();

    for h in longest_chain.iter() {
        remaining_nodes.remove(&h.id());
    }

    // let mut chains: Vec<Vec<Handle>> = vec![longest_chain];

    let mut chains = Chains::default();

    // chains.push_chain(

    //     let n = h.id();
    // })

    let mut running = true;

    let mut chains_by_len = chain_map.by_longest(&graph);

    let mut cur_nodes = Vec::new();

    let mut count = 0;

    while running {
        let node_count = remaining_nodes.len();
        println!("{}", count);

        let mut delete_nodes = false;

        if let Some(((path, chain_ix), len)) = chains_by_len.first() {
            println!("removing chain with length {}", len);
            if let Some(chain) = chain_map.remaining.get_mut(path) {
                if let Some(chain) = chain.remaining.get(*chain_ix) {
                    cur_nodes.clear();
                    cur_nodes.extend(chain.iter().map(|h| h.id()));

                    println!("gathered {} nodes to remove", cur_nodes.len());
                    for node in cur_nodes.iter() {
                        remaining_nodes.remove(node);
                    }
                }

                delete_nodes = true;
                // remove_head = true;
            }
            // let nodes = longest.
        } else {
            println!("no remaining chains");
            // running = false;
            break;
        }

        if delete_nodes {
            for (path_id, chains) in chain_map.remaining.iter_mut() {
                chains.delete_nodes(&cur_nodes);
            }
        }

        count += 1;

        chain_map.by_longest_mut(&graph, &mut chains_by_len);
        println!("  - nodes left {}", node_count);
    }

    let node_count = remaining_nodes.len();
    println!("leftover node count {}", node_count);

    let mut len_map: FxHashMap<usize, usize> = FxHashMap::default();

    for node in remaining_nodes {
        let len = graph.node_len(Handle::pack(node, false));
        *len_map.entry(len).or_default() += 1;

        if len > 100 {
            println!("node {} is long", node.0);
        }
    }

    let mut keys = len_map.keys().collect::<Vec<_>>();
    keys.sort();

    for key in keys {
        println!("{} - {}", key, len_map.get(&key).unwrap());
    }
}

pub struct ChainMap {
    remaining: FxHashMap<PathId, PathChains>,
}

impl ChainMap {
    pub fn new(graph: &PackedGraph, path_pos: &PathPositionMap) -> Self {
        let mut remaining = FxHashMap::default();

        for path in graph.path_ids() {
            let chains = PathChains::from_path(graph, path_pos, path).unwrap();
            remaining.insert(path, chains);
        }

        Self { remaining }
    }

    pub fn remove_path(&mut self, path: PathId) -> Option<PathChains> {
        self.remaining.remove(&path)
    }

    pub fn by_longest_mut(
        &self,
        graph: &PackedGraph,
        res: &mut Vec<((PathId, usize), usize)>,
    ) {
        res.clear();
        res.extend(self.remaining.iter().flat_map(|(path, chains)| {
            chains.lengths(graph).map(|(ix, len)| ((*path, ix), len))
        }));

        res.sort_by_key(|(_, l)| *l);
    }

    // the (PathId, usize) tuple is the index (path + chain index)
    pub fn by_longest(
        &self,
        graph: &PackedGraph,
    ) -> Vec<((PathId, usize), usize)> {
        let mut all_chains = Vec::new();
        self.by_longest_mut(graph, &mut all_chains);
        all_chains
    }
}

pub struct PathChains {
    remaining: Vec<Vec<Handle>>,
}

impl PathChains {
    pub fn lengths<'a>(
        &'a self,
        graph: &'a PackedGraph,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.remaining.iter().enumerate().map(|(ix, chain)| {
            (ix, chain.iter().map(|h| graph.node_len(*h)).sum())
        })
    }

    pub fn length_order(&self, graph: &PackedGraph) -> Vec<(usize, usize)> {
        let mut res: Vec<(usize, usize)> = self
            .remaining
            .iter()
            .enumerate()
            .map(|(ix, chain)| {
                (ix, chain.iter().map(|h| graph.node_len(*h)).sum())
            })
            .collect::<Vec<_>>();

        res.sort_by_key(|(_, l)| *l);
        res
    }

    pub fn from_path(
        graph: &PackedGraph,
        path_pos: &PathPositionMap,
        path: PathId,
    ) -> Option<Self> {
        let steps = graph.path_steps(path)?;
        // let steps = path_pos_steps(graph, path_pos, path)?;
        // let steps = path_pos_steps(graph, path_pos, path)?;

        let mut remaining = Vec::new();

        // for (h, step_ix, pos) in steps {
        for step in steps {
            remaining.push(step.handle());
        }

        let remaining = vec![remaining];

        Some(Self { remaining })
    }

    pub fn delete_nodes(&mut self, nodes: &[NodeId]) {
        let mut to_keep: FxHashMap<usize, Vec<std::ops::Range<usize>>> =
            FxHashMap::default();

        for (ix, chain) in self.remaining.iter().enumerate() {
            let mut ranges_to_keep: Vec<std::ops::Range<usize>> = Vec::new();
            let mut to_keep_start: Option<usize> = None;
            let mut prev_ix: Option<usize> = None;

            for (ix, h) in chain.iter().enumerate() {
                if nodes.contains(&h.id()) {
                    if let Some(start) = to_keep_start {
                        if let Some(prev) = prev_ix {
                            ranges_to_keep.push(start..prev);
                            to_keep_start = None;
                        } else {
                            unreachable!();
                        }
                    } //else {
                      // }
                } else {
                    if to_keep_start.is_none() {
                        to_keep_start = Some(ix);
                    }
                }

                prev_ix = Some(ix);
            }

            to_keep.insert(ix, ranges_to_keep);
        }

        let mut new_remaining: Vec<Vec<Handle>> = Vec::new();

        for (ix, ranges) in to_keep {
            let chain = &self.remaining[ix];

            for range in ranges {
                let new_chain = Vec::from_iter(chain[range].iter().copied());
                new_remaining.push(new_chain);
            }
        }

        self.remaining = new_remaining;
    }
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
        let mut path_ranges: Vec<Vec<std::ops::Range<usize>>> =
            vec![Vec::new(); path_count];
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

        let path_ranges: FxHashMap<PathId, Vec<std::ops::Range<usize>>> =
            path_ranges
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

/*
pub fn path_steps(
    graph: &PackedGraph,
    path_pos: &PathPositionMap,
    path_id: PathId,
) -> Option<Vec<Handle>> {
    let path_steps = graph.path_steps(path_id)?;

    let mut result = Vec::new();

    for step in path_steps {
        let step_ptr = step.0;
        let handle = step.handle();

        let base_pos = path_pos.path_step_position(path_id, step_ptr)?;

        result.push((handle, step_ptr, base_pos));
    }

    Some(result)
}

*/
