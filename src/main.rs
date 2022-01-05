use flexi_logger::{Duplicate, FileSpec, Logger};
use gfa::{gfa::GFA, optfields::OptFields};
use gfa2obj::sparse::GetVectorElementList;
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

use std::{collections::VecDeque, sync::Arc};

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

fn main__() {
    use std::time::Instant;
    let args: Args = argh::from_env();

    let t0 = Instant::now();
    let (adj_mat, segments) =
        gfa2obj::sparse_graph::gfa_to_adj_matrix(&args.gfa_path).unwrap();
    eprintln!("loaded in {} s", t0.elapsed().as_secs_f64());

    // let start = NodeId::from(1);
    let mut single_nucleotide_nodes: Vec<NodeId> = Vec::new();

    for (&id, &len) in segments.iter() {
        if len == 1 {
            let id = NodeId::from(id);
            single_nucleotide_nodes.push(id);
        }
    }

    let limit = 2;

    let snn_count = single_nucleotide_nodes.len();
    eprintln!("number of SNNs: {}", snn_count);

    let mut result_count: Vec<usize> =
        Vec::with_capacity(single_nucleotide_nodes.len());

    let mut insert_count = 0;
    let mut removed_count = 0;

    let mut removed: FxHashSet<NodeId> = FxHashSet::default();

    let t0 = Instant::now();
    for (ix, id) in single_nucleotide_nodes.into_iter().enumerate() {
        if ix % 1000 == 0 {
            eprintln!(" {:7} / {:7}", ix, snn_count);
        }

        let bfs_result =
            gfa2obj::sparse_graph::bfs(&adj_mat, id, Some(limit)).unwrap();

        let n = bfs_result.number_of_stored_elements().unwrap();

        let el_list = bfs_result.get_element_list().unwrap();

        removed.insert(id);

        for ix_ in el_list.indices_ref() {
            // for val in  el_list.values_ref() {
            let node_id = NodeId::from(ix_ + 1);
            removed.insert(node_id);
        }

        result_count.push(n);

        // removed_count += n;
        insert_count += 1;
    }

    insert_count /= 2;

    eprintln!(
        "limit {} BFSes across all {} SNNs took {} s",
        limit,
        snn_count,
        t0.elapsed().as_secs_f64()
    );

    eprintln!("removed.len(): {}", removed.len());
    eprintln!("insert_count: {}", insert_count);

    eprintln!("graph node count: {}", segments.len());

    let s = segments.len() as isize;
    let r = removed.len() as isize;
    let i = insert_count as isize;

    eprintln!("|nodes| - |removed| + |inserted| = {}", s - r + i);
}

#[derive(Clone)]
pub struct Chain {
    chain: VecDeque<NodeId>,
    left: FxHashSet<NodeId>,
    right: FxHashSet<NodeId>,
}

#[derive(Clone)]
pub struct ChainComplex {
    chains: Vec<Chain>,

    lens: Vec<usize>,

    chains_children: Vec<Vec<(usize, std::ops::RangeInclusive<usize>)>>,
    chains_parent: Vec<Option<usize>>,
}

pub struct Curve {
    total_length: usize,

    /// each chain/curve has a map from nodes contained in the chain
    /// to the position offset from the start, and node length
    node_pos_offsets: FxHashMap<NodeId, (usize, usize)>,
    // child_anchor_f32: Vec<(f32, f32)>,
    // child_anchor_ix: Vec<std::ops::Range<usize>>,
    // child_anchor_id: Vec<(NodeId, NodeId)>,
}

pub struct CurveComplex {
    chain_complex: Arc<ChainComplex>,

    curves: Vec<Curve>,
}

impl CurveComplex {
    pub fn chain_complex_mut(&mut self) -> &mut ChainComplex {
        Arc::make_mut(&mut self.chain_complex)
    }

    pub fn from_chain_complex(
        graph: &PackedGraph,
        chain_complex: ChainComplex,
    ) -> Self {
        let chain_complex = Arc::new(chain_complex);
        let mut curves = Vec::new();

        for chain_ix in 0..chain_complex.chains.len() {
            let chain = &chain_complex.chains[chain_ix];

            let children = &chain_complex.chains_children[chain_ix];

            let chain_len = chain_complex.lens[chain_ix];

            let mut offset = 0;

            let mut node_pos_offsets: FxHashMap<NodeId, (usize, usize)> =
                FxHashMap::default();

            // 1. step through the entire chain and find the bp pos
            // for each node
            for (ix, node) in chain.chain.iter().enumerate() {
                let h = Handle::pack(*node, false);
                let node_len = graph.node_len(h);

                // 2. for each child, map the indices to curve offsets
                // in the range [0..1];
                node_pos_offsets.insert(*node, (offset, node_len));

                offset += node_len;
            }

            let total_length = offset;
            assert!(total_length == chain_len);

            let curve = Curve {
                total_length,
                node_pos_offsets,
            };

            curves.push(curve);
        }

        Self {
            chain_complex,
            curves,
        }
    }
}

impl ChainComplex {
    pub fn from_chains(graph: &PackedGraph, chains: Vec<Chain>) -> Self {
        let chains_children = vec![Vec::new(); chains.len()];
        let chains_parent = vec![None; chains.len()];

        let mut max_len = 0;

        let mut chains_lens: Vec<_> = chains
            .into_iter()
            .map(|chain| {
                let len = chain
                    .chain
                    .iter()
                    .map(|n| graph.node_len(Handle::pack(*n, false)))
                    .sum::<usize>();

                max_len = max_len.max(len);

                (chain, len)
            })
            .collect();

        chains_lens.sort_by_key(|(_, l)| *l);
        chains_lens.reverse();

        let (chains, lens) = chains_lens.into_iter().unzip();

        Self {
            chains,
            lens,
            chains_children,
            chains_parent,
        }
    }

    /// returns false if the child already had a parent, but does not
    /// check whether the child is valid
    pub fn attach_child(&mut self, parent: usize, child: usize) -> bool {
        if self.chains_parent[child].is_some() {
            return false;
        }

        let p = &self.chains[parent];
        let c = &self.chains[child];

        let mut left: Option<usize> = None;
        let mut right: Option<usize> = None;

        for (ix, node) in p.chain.iter().enumerate() {
            if c.left.contains(node) {
                left = Some(ix);
            }
            if c.right.contains(node) {
                right = Some(ix);
            }

            if left.is_some() && right.is_some() {
                break;
            }
        }

        if let (Some(left), Some(right)) = (left, right) {
            let range = left..=right;

            let children = &mut self.chains_children[parent];
            children.push((child, range));

            self.chains_parent[child] = Some(parent);

            true
        } else {
            false
        }
    }

    pub fn potential_children_for(&self, parent: usize) -> Vec<usize> {
        let mut res = Vec::new();

        let chain_nodes = self.chains[parent]
            .chain
            .iter()
            .copied()
            .collect::<FxHashSet<_>>();

        for (ix, other) in self.chains.iter().enumerate() {
            if ix == parent || self.chains_parent[ix].is_some() {
                continue;
            }

            let found_left = !other.left.is_disjoint(&chain_nodes);
            let found_right = !other.right.is_disjoint(&chain_nodes);

            if found_left && found_right {
                res.push(ix);
            }
        }

        res
    }
}

fn main() {
    // let spec = match (args.trace, args.debug, args.quiet) {
    //     (true, _, _) => "trace",
    //     (_, true, _) => "debug",
    //     (_, _, true) => "",
    //     _ => "info",
    // };

    let logger = Logger::try_with_env_or_str("debug")
        .unwrap()
        .log_to_file(FileSpec::default())
        .duplicate_to_stderr(Duplicate::Debug)
        .start()
        .unwrap();

    use std::time::Instant;

    let args: Args = argh::from_env();

    eprintln!("loading graph");

    let t0 = Instant::now();
    let (graph, path_pos) = load_gfa(&args.gfa_path).unwrap();
    eprintln!("loaded in {} s", t0.elapsed().as_secs_f64());

    let mut stack: VecDeque<Handle> = VecDeque::new();

    let mut remaining_nodes =
        graph.handles().map(|h| h.id()).collect::<FxHashSet<_>>();

    // let mut open_left = true;
    // let mut open_right = true;
    let mut current_chain: VecDeque<NodeId> = VecDeque::new();

    let mut visited: FxHashSet<NodeId> = FxHashSet::default();

    // let mut chains: Vec<VecDeque<NodeId>> = Vec::new();
    let mut chains: Vec<Chain> = Vec::new();

    let mut count = 0;

    // let mut chain_complex = ChainComplex {
    //     chains: Vec::new(),
    //     chain_left_anchors: Vec::new(),
    //     chain_right_anchors: Vec::new(),
    // };

    loop {
        if let Some(&node) = remaining_nodes.iter().next() {
            if visited.contains(&node) {
                remaining_nodes.remove(&node);
                continue;
            }

            current_chain.push_back(node);

            let handle = Handle::pack(node, false);
            let rev = handle.flip();

            stack.push_back(handle);
            stack.push_front(rev);

            while let Some(cur) = stack.pop_back() {
                remaining_nodes.remove(&cur.id());
                visited.insert(cur.id());

                if cur.is_reverse() {
                    current_chain.push_front(cur.id());
                } else {
                    current_chain.push_back(cur.id());
                }

                let mut fwd_n = graph
                    .neighbors(cur, Direction::Right)
                    .filter(|h| !visited.contains(&h.id()))
                    .collect::<Vec<_>>();

                // continue with the longest neighboring node
                fwd_n.sort_by_key(|&h| graph.node_len(h));
                fwd_n.first().map(|&h| stack.push_back(h));
            }

            let nbors = |iter: &mut dyn Iterator<Item = &&NodeId>, rev| {
                iter.flat_map(|&&n| {
                    graph.neighbors(Handle::pack(n, rev), Direction::Right)
                })
                .map(|h| h.id())
                .collect()
            };

            let left = nbors(&mut current_chain.back().iter(), true);
            let right = nbors(&mut current_chain.front().iter(), false);

            let chain = Chain {
                chain: std::mem::take(&mut current_chain),
                left,
                right,
            };

            chains.push(chain);
        } else {
            break;
        }

        count += 1;
    }

    let mut chain_complex = ChainComplex::from_chains(&graph, chains);
    {
        let longest_ix = 0;
        let longest = &chain_complex.lens[0];

        eprintln!(
            "{:?}\t{:?}",
            &chain_complex.chains[longest_ix].left,
            &chain_complex.chains[longest_ix].right
        );
        eprintln!(
            "finished with {} chains after {} iterations",
            chain_complex.chains.len(),
            count
        );

        eprintln!("longest chain: {}", longest);
    }

    for parent in 0..chain_complex.chains.len() {
        let pot_children = chain_complex.potential_children_for(parent);

        for child in pot_children {
            let result = chain_complex.attach_child(parent, child);
            if result {
                eprintln!("attached parent {} - child {}", parent, child);
            }
        }
    }

    use rand::prelude::*;

    let mut rng = thread_rng();

    let mut visited: FxHashSet<NodeId> = FxHashSet::default();

    for (chain, parent) in chain_complex
        .chains
        .iter()
        .zip(chain_complex.chains_parent.iter())
    {
        let r = rng.gen::<u8>();
        let g = rng.gen::<u8>();
        let b = rng.gen::<u8>();
        let a = 255u8;

        for (ix, node) in chain.chain.iter().enumerate() {
            if ix == 0 && parent.is_none() {
                println!("{}\t{}", node.0, node.0);
            }

            if !visited.insert(*node) {
                continue;
            }

            println!("{}\t#{:2x}{:2x}{:2x}{:2x}", node.0, r, g, b, a);
        }
    }
}

fn main_() {
    use std::time::Instant;

    let args: Args = argh::from_env();

    eprintln!("loading graph");

    let t0 = Instant::now();
    let (graph, path_pos) = load_gfa(&args.gfa_path).unwrap();
    eprintln!("loaded in {} s", t0.elapsed().as_secs_f64());

    let mut stack: VecDeque<NodeId> = VecDeque::new();

    let mut pass = 0;

    let mut passes: Vec<Vec<NodeId>> = Vec::new();

    let mut remaining_nodes =
        graph.handles().map(|h| h.id()).collect::<FxHashSet<_>>();

    let mut remaining_order =
        remaining_nodes.iter().copied().collect::<Vec<_>>();
    // remaining_order.sort();
    // remaining_order.reverse();
    // graph.handles().map(|h| h.id()).collect::<FxHashSet<_>>();

    let mut len_map: FxHashMap<usize, usize> = FxHashMap::default();

    let total_t = Instant::now();
    loop {
        // let mut end_pass = false;
        // for handle in graph.handles() {
        // while let Some(handle) = graph
        //     // for handle in graph
        //     .handles()
        //     .filter(|h| remaining_nodes.contains(&h.id()))
        //     .next()

        /*
        for (node, mark_done) in remaining_nodes.iter().zip(marks.iter_mut()) {
            //
        }
        */

        // if let Some(node) = remaining_order.pop() {
        while let Some(node) = remaining_order.pop() {
            // let node = handle.id();

            if !remaining_nodes.contains(&node) {
                continue;
            }

            stack.push_back(node);

            // println!("{}", remaining_nodes.len());

            let mut this_pass = Vec::new();

            let t0 = Instant::now();

            while let Some(current) = stack.pop_back() {
                remaining_nodes.remove(&current);
                this_pass.push(current);

                let handle = Handle::pack(current, false);

                if let Some(next) = graph
                    .neighbors(handle, Direction::Right)
                    // .chain(graph.neighbors(handle, Direction::Left))
                    .filter(|other| remaining_nodes.contains(&other.id()))
                    .next()
                {
                    stack.push_back(next.id());
                }

                /*
                if let Some(next) = graph
                    .neighbors(handle, Direction::Right)
                    .chain(graph.neighbors(handle, Direction::Left))
                    .filter(|other| remaining_nodes.contains(&other.id()))
                    .next()
                {
                    stack.push_back(next.id());
                }
                */
                // else {
                //     end_pass = true;
                //     break;
                // }
            }

            let pass_len = this_pass
                .iter()
                .map(|n| graph.node_len(Handle::pack(*n, false)))
                .sum::<usize>();

            if pass < 30 {
                eprintln!(
                    "pass {}\t {} left\tlen: {}",
                    pass,
                    remaining_nodes.len(),
                    pass_len
                );
            }

            *len_map.entry(pass_len).or_default() += 1;

            passes.push(this_pass);
            pass += 1;
            // if end_pass {
            //     break;
            // }
        }

        remaining_order.clear();
        remaining_order.extend(remaining_nodes.iter().copied());
        remaining_order.sort();
        remaining_order.reverse();

        // println!("pass {} took {} s", pass, t0.elapsed().as_secs_f64());

        if remaining_nodes.is_empty() {
            break;
        }
    }

    eprintln!("took {} s", total_t.elapsed().as_secs_f64());

    eprintln!("done in {} passes", pass);

    // let mut total_len = 0;
    let total_len = passes
        .iter()
        .map(|nodes| {
            nodes
                .iter()
                .map(|n| graph.node_len(Handle::pack(*n, false)))
                .sum::<usize>()
        })
        .sum::<usize>();

    let avg_len = (total_len as f64) / (pass as f64);

    eprintln!("avg len: {}", avg_len);

    let mut len_vec = len_map.into_iter().collect::<Vec<_>>();
    len_vec.sort_by_key(|(_len, count)| *count);

    // let limit = 100;
    // let limit = 10;
    // let limit = 4;
    // let limit = 2;
    let limit = 1;

    let mut filtered_passes = len_vec
        .iter()
        .filter_map(
            |(len, count)| if *len < limit { Some(*count) } else { None },
        )
        .collect::<Vec<_>>();

    let filtered_count = filtered_passes.iter().sum::<usize>();

    eprintln!("passes with len < {}: {}", limit, filtered_count);
    eprintln!("passes with len > {}: {}", limit, pass - filtered_count);

    let filtered_passes: Vec<_> = passes
        .into_iter()
        .filter(|pass| {
            let len = pass
                .iter()
                .map(|n| graph.node_len(Handle::pack(*n, false)))
                .sum::<usize>();

            len >= limit
        })
        .collect::<Vec<_>>();
    eprintln!("built filtered passes");

    let pass_nbors: Vec<(FxHashSet<NodeId>, FxHashSet<NodeId>)> =
        filtered_passes
            .iter()
            .map(|pass| {
                let start = *pass.first().unwrap();
                let end = *pass.last().unwrap();

                let start_n = graph
                    .neighbors(Handle::pack(start, false), Direction::Left)
                    .map(|h| h.id())
                    .collect::<FxHashSet<_>>();
                let end_n = graph
                    .neighbors(Handle::pack(start, false), Direction::Right)
                    .map(|h| h.id())
                    .collect::<FxHashSet<_>>();

                (start_n, end_n)
            })
            .collect();

    let mut done: FxHashSet<(usize, usize)> = FxHashSet::default();

    let mut singles = 0;
    let mut doubles = 0;

    let maxx = pass_nbors.len();

    let mut new_passes: Vec<Vec<NodeId>> = Vec::new();

    let mut cur_pass: Vec<NodeId> = Vec::new();

    let mut current_neighbors: FxHashSet<NodeId> = FxHashSet::default();

    let mut remaining_indices = (0..pass_nbors.len()).collect::<FxHashSet<_>>();

    while let Some(ix) = remaining_indices.iter().next().copied() {
        remaining_indices.remove(&ix);

        let mut to_remove: Vec<usize> = Vec::new();

        let (out_l, out_r) = pass_nbors.get(ix).unwrap();
        let this_pass = filtered_passes.get(ix).unwrap();

        cur_pass.clear();
        cur_pass.extend(this_pass.iter().copied());

        current_neighbors.clear();
        let last = *this_pass.last().unwrap();
        let handle = Handle::pack(last, false);
        current_neighbors
            .extend(graph.neighbors(handle, Direction::Right).map(|h| h.id()));

        let mut inner_iter = remaining_indices.iter().copied();

        while let Some(ix_) = inner_iter.next() {
            if ix_ != ix {
                let (in_l, in_r) = pass_nbors.get(ix_).unwrap();
                let other_pass = filtered_passes.get(ix_).unwrap();

                if !out_r.is_disjoint(in_l) {
                    cur_pass.extend(other_pass.iter().copied());
                    current_neighbors.clear();
                    current_neighbors.extend(in_r.iter().copied());
                    to_remove.push(ix_);
                }
            }
        }

        let pass = std::mem::take(&mut cur_pass);

        new_passes.push(pass);

        for to_rem in to_remove {
            remaining_indices.remove(&to_rem);
        }
    }

    println!("new_passes.len() {}", new_passes.len());

    let mut total_len_ = 0;

    let mut snp_len = 0;

    let mut by_len = Vec::new();

    for (ix, pass) in new_passes.iter().enumerate() {
        let mut pass_len = 0;
        for &node in pass.iter() {
            let len = graph.node_len(Handle::pack(node, false));

            if len == 1 && pass.len() == 1 {
                snp_len += 1;
            } else {
                total_len_ += len;
                pass_len += len;
            }
        }

        by_len.push((ix, pass_len));
        // let mut pass_len = 0;
    }

    by_len.sort_by_key(|(_, l)| *l);

    println!("graph len: {}", graph.total_length());
    println!("total len: {}", total_len_);
    println!("snp len: {}", snp_len);

    println!("----------");

    let mut another_map: FxHashMap<Handle, usize> = FxHashMap::default();

    for (ix, len) in by_len {
        let pass = &new_passes[ix];

        let left = pass.first().unwrap();
        let right = pass.last().unwrap();

        let left_adj = graph
            .neighbors(Handle::pack(*left, false), Direction::Left)
            // .chain(
            //     graph.neighbors(Handle::pack(*left, false), Direction::Right),
            // )
            .map(|h| {
                *another_map.entry(h).or_default() += 1;
                h.0
            })
            .collect::<Vec<_>>();
        let right_adj = graph
            .neighbors(Handle::pack(*right, false), Direction::Right)
            // .chain(
            //     graph.neighbors(Handle::pack(*right, false), Direction::Left),
            // )
            // .map(|h| h.0)
            .map(|h| {
                *another_map.entry(h).or_default() += 1;
                h.0
            })
            .collect::<Vec<_>>();

        println!("{:4} - {:6}\t{:?} - {:?}", ix, len, left_adj, right_adj);
    }

    println!("------------------");

    for (h, count) in another_map {
        if count > 1 {
            println!("{:?}\t{}", h, count);
        }
    }
    // println!("{:?}", another_map);

    // let mut used_indices: FxHashSet<usize> = FxHashSet::default();

    /*
    for (ix, ((out_l, out_r), this_pass)) in
        pass_nbors.iter().zip(filtered_passes.iter()).enumerate()
    {
        //
        if used_indices.contains(&ix) {
            continue;
        }

        cur_pass.clear();
        cur_pass.extend(this_pass.iter().copied());

        current_neighbors.clear();
        let last = *this_pass.last().unwrap();
        let handle = Handle::pack(last, false);
        current_neighbors
            .extend(graph.neighbors(handle, Direction::Right).map(|h| h.id()));
    }
    */

    /*
    for (ix, (out_l, out_r)) in pass_nbors.iter().enumerate() {
        println!(" - {} \t / {}", ix, maxx);
        for (ix_, (in_l, in_r)) in pass_nbors.iter().enumerate() {
            let lo = ix.min(ix_);
            let hi = ix.max(ix_);

            if done.contains(&(lo, hi)) || done.contains(&(hi, lo)) || lo == hi
            {
                continue;
            }

            let left_match = !out_l.is_disjoint(in_l);
            let right_match = !out_r.is_disjoint(in_r);

            // let op_left_match = !out_l.is_disjoint(in_r);
            // let op_right_match = !out_r.is_disjoint(in_l);

            if left_match && right_match {
                singles += 1;
            } else if left_match || right_match {
                doubles += 1;
            }

            done.insert((lo, hi));
        }
    }
    */

    println!("total: {}", pass);
    println!("doubles: {}", doubles);
    println!("singles: {}", singles);

    // for (ix, pass) in passes.into_iter().enumerate() {
    //
    // }
}

/*
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
*/

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
        // let mut to_keep: FxHashMap<usize, Vec<std::ops::Range<usize>>> =
        type ChainRange =
            (Option<Handle>, Option<Handle>, std::ops::Range<usize>);

        let mut to_keep: FxHashMap<usize, Vec<ChainRange>> =
            FxHashMap::default();

        for (ix, chain) in self.remaining.iter().enumerate() {
            let mut ranges_to_keep: Vec<ChainRange> = Vec::new();
            let mut to_keep_start: Option<usize> = None;
            let mut prev_ix: Option<usize> = None;

            let mut left_handle: Option<Handle> = None;
            let mut cur_handle: Option<Handle> = None;

            for (ix, h) in chain.iter().enumerate() {
                if nodes.contains(&h.id()) {
                    if let Some(start) = to_keep_start {
                        if let Some(prev) = prev_ix {
                            let left = left_handle;
                            // let left = todo!();
                            let right = cur_handle;
                            // let right = todo!();
                            let range = start..prev;
                            ranges_to_keep.push((left, right, range));
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

            for (left, right, range) in ranges {
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
