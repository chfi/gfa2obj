use std::{io::BufReader, sync::Arc};

use crate::sparse::*;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use std::sync::Mutex;

use lazy_static::lazy_static;

use anyhow::Result;

lazy_static! {
    static ref GRAPHBLAS_CTX: Arc<Context> =
        Context::init_ready(Mode::NonBlocking).unwrap();
}

pub fn gfa_to_adj_matrix(gfa_path: &str) -> Result<SparseMatrix<bool>> {
    use std::fs::File;
    use std::io::prelude::*;

    // use bstr::ByteSlice;

    let file = File::open(gfa_path)?;
    let mut reader = BufReader::new(file);

    let mut buf = String::with_capacity(1024);

    let mut segments: FxHashMap<usize, usize> = FxHashMap::default();
    let mut links: FxHashSet<(usize, usize)> = FxHashSet::default();

    loop {
        buf.clear();
        let readn = reader.read_line(&mut buf)?;

        if readn == 0 {
            break;
        }

        let line = &buf[..readn];

        let mut fields = line.split("\t");

        match fields.next() {
            Some("S") => {
                // let id = fields.next().map(|f| f.parse::<usize>()).flatten();
                let id_str = fields.next().unwrap();
                let id = id_str.parse::<usize>()?;

                let seq = fields.next().unwrap();
                segments.insert(id, seq.len());
            }
            Some("L") => {
                let id_a_str = fields.next().unwrap();
                let id_a = id_a_str.parse::<usize>()?;

                let _orient = fields.next();

                let id_b_str = fields.next().unwrap();
                let id_b = id_b_str.parse::<usize>()?;

                let left = id_a.min(id_b);
                let right = id_a.max(id_b);

                links.insert((left, right));
            }
            _ => (),
        }
    }

    let n = segments.len();

    let matrix_size = Size::new(n, n);
    let mut adj_matrix: SparseMatrix<bool> =
        SparseMatrix::new(&GRAPHBLAS_CTX, &matrix_size)?;

    for link in links {
        let row = link.0 - 1;
        let column = link.1 - 1;
        adj_matrix
            .set_element(MatrixElement::from_triple(row, column, true))?;
    }

    Ok(adj_matrix)
}

fn parallel_calls_to_graphblas() {
    let context = Context::init_ready(Mode::NonBlocking).unwrap();

    let number_of_matrices = 100;

    let matrix_size = Size::new(10, 5);
    let mut matrices: Vec<SparseMatrix<i32>> = (0..number_of_matrices)
        .into_par_iter()
        .map(|_| SparseMatrix::<i32>::new(&context, &matrix_size).unwrap())
        .collect();

    matrices.par_iter_mut().for_each(|matrix| {
        matrix
            .set_element(MatrixElement::from_triple(1, 2, 3))
            .unwrap()
    });

    let add_operator = Plus::<i32, i32, i32>::new();
    let options = OperatorOptions::new_default();
    let result_matrix =
        Mutex::new(SparseMatrix::<i32>::new(&context, &matrix_size).unwrap());

    let element_wise_matrix_add_operator =
        EwiseMatMulBinOp::<i32, i32, i32>::new(
            &add_operator,
            &options,
            Some(&add_operator),
        );

    matrices.par_iter().for_each(|matrix| {
        element_wise_matrix_add_operator
            .apply(&matrix, &matrix, &mut result_matrix.lock().unwrap())
            .unwrap();
    });

    let result_matrix = result_matrix.into_inner().unwrap();

    assert_eq!(
        600,
        result_matrix
            .get_element(Coordinate::new(1, 2))
            .unwrap()
            .value()
    );
}
