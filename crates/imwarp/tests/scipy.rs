//! `map_coordinates` against SciPy output (`tools/make_goldens.py`).

use imwarp::{BoundaryMode, Interpolation, Order, map_coordinates};
use ndarray::Array2;
use serde_json::Value;

const MODES: [(&str, BoundaryMode); 5] = [
    ("constant", BoundaryMode::Constant),
    ("nearest", BoundaryMode::Nearest),
    ("mirror", BoundaryMode::Mirror),
    ("reflect", BoundaryMode::Reflect),
    ("wrap", BoundaryMode::Wrap),
];
const ORDERS: [(u8, Order); 3] = [(0, Order::Nearest), (1, Order::Linear), (3, Order::Cubic)];
const RTOL: f64 = 1e-11;
const ATOL: f64 = 1e-11;

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= ATOL + RTOL * w.abs();
        assert!(ok, "{label}[{i}]: got {g}, want {w}");
    }
}

fn sample(image: &Array2<f64>, coords: &[f64], options: &Interpolation) -> Vec<f64> {
    // `coords` is (2, N), row-major; lay the points out as a 1 x N image.
    let n = coords.len() / 2;
    let rows = Array2::from_shape_fn((1, n), |(_, i)| coords[i]);
    let cols = Array2::from_shape_fn((1, n), |(_, i)| coords[n + i]);
    map_coordinates(image.view(), rows.view(), cols.view(), options)
        .expect("valid arguments")
        .iter()
        .copied()
        .collect()
}

#[test]
fn matches_scipy_for_every_order_mode_and_shape() {
    let golden: Value =
        serde_json::from_str(include_str!("data/scipy.json")).expect("the golden file");
    let mut compared = 0;
    for case in golden["cases"].as_array().expect("cases") {
        let shape = numbers(&case["shape"]);
        let (ny, nx) = (shape[0] as usize, shape[1] as usize);
        let data = Array2::from_shape_vec((ny, nx), numbers(&case["data"])).expect("data");
        let coords = numbers(&case["coords"]);
        let expected = &case["expected"];
        for (o, order) in ORDERS {
            for (name, mode) in MODES {
                let got = sample(&data, &coords, &Interpolation::new(order, mode, -7.5));
                let key = format!("o{o}_{name}");
                assert_close(
                    &format!("{key} {:?}", data.dim()),
                    &got,
                    &numbers(&expected[&key]),
                );
                compared += 1;
            }
        }
        let nan_fill = sample(
            &data,
            &coords,
            &Interpolation::new(Order::Linear, BoundaryMode::Constant, f64::NAN),
        );
        assert_close(
            "nan cval",
            &nan_fill,
            &numbers(&expected["o1_constant_nan"]),
        );
        let unfiltered =
            Interpolation::new(Order::Linear, BoundaryMode::Constant, 0.0).with_prefilter(false);
        assert_close(
            "o1 noprefilter",
            &sample(&data, &coords, &unfiltered),
            &numbers(&expected["o1_noprefilter_constant"]),
        );
        let cubic_raw =
            Interpolation::new(Order::Cubic, BoundaryMode::Mirror, 0.0).with_prefilter(false);
        assert_close(
            "o3 noprefilter",
            &sample(&data, &coords, &cubic_raw),
            &numbers(&expected["o3_noprefilter_mirror"]),
        );
    }
    assert_eq!(compared, 75);
}

#[test]
fn rejects_bad_arguments() {
    let image = Array2::<f64>::zeros((3, 3));
    let a = Array2::<f64>::zeros((2, 2));
    let b = Array2::<f64>::zeros((2, 3));
    let opts = Interpolation::default();
    assert!(
        map_coordinates(image.view(), a.view(), b.view(), &opts).is_err(),
        "mismatched coordinate shapes"
    );
    assert!(
        map_coordinates(
            Array2::<f64>::zeros((0, 3)).view(),
            a.view(),
            a.view(),
            &opts
        )
        .is_err(),
        "empty image"
    );
    assert!(Order::from_u8(2).is_err());
    assert_eq!(Order::from_u8(3).unwrap(), Order::Cubic);
}

#[test]
fn nan_coordinates_take_the_fill_value() {
    let image = Array2::from_elem((3, 3), 1.0);
    let nan = Array2::from_elem((1, 1), f64::NAN);
    let zero = Array2::zeros((1, 1));
    let got = map_coordinates(
        image.view(),
        nan.view(),
        zero.view(),
        &Interpolation::new(Order::Linear, BoundaryMode::Reflect, -2.0),
    )
    .unwrap();
    assert_eq!(got[[0, 0]], -2.0);
}
