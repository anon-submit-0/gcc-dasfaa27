-- Grounded Consumption Contracts: anonymized reviewer DDL.
-- The three tables govern generated factual claims at the consumption boundary.

CREATE TABLE IF NOT EXISTS gov_grounding_claim (
  claim_id            VARCHAR(64)  NOT NULL,
  claim_type          VARCHAR(32)  NOT NULL,
  logical_field       VARCHAR(128) NOT NULL,
  freshness_clock     VARCHAR(64),
  min_evidence_level  VARCHAR(16)  NOT NULL,
  disposition_policy  VARCHAR(32)  NOT NULL,
  graph_version       VARCHAR(32)  NOT NULL,
  PRIMARY KEY (claim_id)
);

CREATE TABLE IF NOT EXISTS gov_grounding_evidence (
  claim_id            VARCHAR(64)  NOT NULL,
  physical_column_fqn VARCHAR(256) NOT NULL,
  freshness_clock     VARCHAR(64),
  min_evidence_level  VARCHAR(16) NOT NULL,
  evidence_rank       INT NOT NULL DEFAULT 1,
  PRIMARY KEY (claim_id, physical_column_fqn)
);

CREATE TABLE IF NOT EXISTS gov_grounding_coverage (
  claim_type          VARCHAR(32) NOT NULL,
  product_grain       VARCHAR(32) NOT NULL,
  covered             INT NOT NULL,
  total               INT NOT NULL,
  evidence_level      VARCHAR(16) NOT NULL,
  last_checked        DATETIME NOT NULL,
  PRIMARY KEY (claim_type, product_grain, last_checked)
);

INSERT INTO gov_grounding_claim
  (claim_id, claim_type, logical_field, freshness_clock, min_evidence_level,
   disposition_policy, graph_version)
VALUES
  ('price',        'price',        'product.effective_price', 'price_validated_at',     'governed', 'refuse',  'v1.0'),
  ('stock',        'stock',        'item.availability_status','inventory_validated_at', 'governed', 'refuse',  'v1.0'),
  ('spec',         'spec',         'product.specs',           NULL,                     'governed', 'caveat',  'v1.0'),
  ('item_code',    'identity',     'item.code',               NULL,                     'governed', 'surface', 'v1.0'),
  ('region',       'stock',        'item.region_code',        'inventory_validated_at', 'governed', 'surface', 'v1.0'),
  ('brand',        'identity',     'product.brand',           NULL,                     'governed', 'surface', 'v1.0'),
  ('name',         'identity',     'product.name',            NULL,                     'governed', 'surface', 'v1.0'),
  ('compat',       'compatibility','product.compatibility',   'valid_to',               'governed', 'refuse',  'v1.0'),
  ('selling_point','selling_point','product.selling_point',   NULL,                     'derived',  'caveat',  'v1.0'),
  ('fit',          'fit',          'session.fit_profile',     'recorded_at',            'derived',  'degrade', 'v1.0');

INSERT INTO gov_grounding_evidence
  (claim_id, physical_column_fqn, freshness_clock, min_evidence_level,
   evidence_rank)
VALUES
  ('price',         'product_base.effective_price',       'product_base.price_validated_at',       'governed', 1),
  ('stock',         'item_inventory.availability_status', 'item_inventory.inventory_validated_at', 'governed', 1),
  ('stock',         'item_inventory.is_sellable',         'item_inventory.inventory_validated_at', 'governed', 2),
  ('spec',          'product_base.specs',                 NULL,                                    'governed', 1),
  ('item_code',     'item_inventory.item_code',           NULL,                                    'governed', 1),
  ('region',        'item_inventory.region_code',         'item_inventory.inventory_validated_at', 'governed', 1),
  ('brand',         'product_base.brand',                 NULL,                                    'governed', 1),
  ('name',          'product_base.name',                  NULL,                                    'governed', 1),
  ('compat',        'product_relation.target_product_id', 'product_relation.valid_to',             'governed', 1),
  ('selling_point', 'product_base.specs',                 NULL,                                    'derived',  1),
  ('fit',           'user_profile_signal.value',          'user_profile_signal.recorded_at',       'derived',  1);

-- Assertion family used by the paper and artifact:
-- G1 groundedness: every surfaced external claim requires governed evidence.
-- G2 price freshness: price evidence must satisfy the declared clock bound.
-- G3 compatibility groundability: compatibility claims require relation rows.
-- G4 sensitivity guard: high-sensitivity profile signals cannot surface.
-- G5 funnel monotonicity: candidate >= eligible >= ranked item sets.
